"""
connec.py — Connectivité écologique
=====================================
Indicateurs simples de connectivité spatiale entre patches d'habitats cœur.

Approche : centroïdes des patches, comptage des voisins à moins d'un seuil,
distance au plus proche voisin. Pas un modèle least-cost path.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix


# ============================================================================
# PARAMÈTRES (modifiables)
# ============================================================================

METRIC_CRS = "EPSG:2154"

THEME_COLS: List[str] = [
    "UrbainDens", "UrbainDiff", "ZoneIndCom", "Routes",
    "Colza", "CerealPail", "Proteagine", "Soja", "Tournesol",
    "Mais", "Riz", "TuberRacin",
    "Prairie", "Vergers", "Vignes",
    "Feuillus", "Coniferes", "Pelouse", "Landes",
    "SurfMin", "PlageDune", "GlaceNeige", "Eau", "Serres",
]

LABEL_FR: Dict[str, str] = {
    "UrbainDens": "Urbain dense",
    "UrbainDiff": "Urbain diffus",
    "ZoneIndCom": "Zone industrielle/commerciale",
    "Routes":     "Routes",
    "Colza":      "Colza",
    "CerealPail": "Céréales/pailles",
    "Proteagine": "Protéagineux",
    "Soja":       "Soja",
    "Tournesol":  "Tournesol",
    "Mais":       "Maïs",
    "Riz":        "Riz",
    "TuberRacin": "Tubercules/racines",
    "Prairie":    "Prairie",
    "Vergers":    "Vergers",
    "Vignes":     "Vignes",
    "Feuillus":   "Feuillus",
    "Coniferes":  "Conifères",
    "Pelouse":    "Pelouse",
    "Landes":     "Landes",
    "SurfMin":    "Surface minérale",
    "PlageDune":  "Plage/dune",
    "GlaceNeige": "Glace/neige",
    "Eau":        "Eau",
    "Serres":     "Serres",
}

DEFAULT_CORE_HABITATS: List[str] = ["Feuillus", "Prairie", "Eau"]


# ============================================================================
# OUTILS INTERNES
# ============================================================================

def _ensure_cover_label(occ: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Garantit la présence d'une colonne 'cover_label'.
    Si absente, reconstruit l'habitat dominant depuis les colonnes thématiques.
    """
    occ = occ.copy()
    if "cover_label" in occ.columns:
        return occ

    missing = [c for c in THEME_COLS if c not in occ.columns]
    if missing:
        raise ValueError(f"Impossible de reconstruire 'cover_label' — colonnes manquantes : {missing}")

    for col in THEME_COLS:
        occ[col] = pd.to_numeric(occ[col], errors="coerce").fillna(0)

    occ["cover_code_max"] = occ[THEME_COLS].idxmax(axis=1)
    occ["cover_label"]    = occ["cover_code_max"].map(LABEL_FR).fillna(occ["cover_code_max"])
    return occ


def _classify_connectivity(value: float, q33: float, q66: float) -> str:
    if value >= q66: return "élevée"
    if value >= q33: return "intermédiaire"
    return "faible"


# ============================================================================
# CALCUL DE CONNECTIVITÉ
# ============================================================================

def calculate_connectivity(
    landcover_file: str,
    threshold_m: float = 1000.0,
    metric_crs: str = METRIC_CRS,
    core_habitats: Optional[Iterable[str]] = None,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Calcule les indicateurs de connectivité spatiale pour les habitats cœur.

    Paramètres
    ----------
    landcover_file  : chemin vers le GeoJSON d'occupation des sols
    threshold_m     : distance max (m) entre centroïdes pour considérer deux patches connectés
    metric_crs      : CRS métrique pour les calculs de distance
    core_habitats   : habitats à inclure dans le réseau (défaut : Feuillus, Prairie, Eau)

    Retour
    ------
    patches_gdf   : GeoDataFrame des patches avec score de connectivité
    summary_df    : synthèse par habitat
    global_stats  : indicateurs globaux
    """
    if core_habitats is None:
        core_habitats = DEFAULT_CORE_HABITATS

    occ = gpd.read_file(landcover_file)
    occ = _ensure_cover_label(occ)

    if occ.empty:
        raise ValueError("Le fichier d'occupation des sols est vide.")

    if occ.crs is None:
        occ = occ.set_crs("EPSG:4326")
    occ = occ.to_crs(metric_crs).copy()
    occ["area_m2"] = occ.geometry.area

    core = occ[occ["cover_label"].isin(list(core_habitats))].copy()
    if core.empty:
        raise ValueError(f"Aucun patch trouvé pour les habitats cœur : {list(core_habitats)}")

    core["patch_id"] = np.arange(len(core))
    core["centroid"] = core.geometry.centroid
    centroids = np.array([[g.x, g.y] for g in core["centroid"]])

    if len(core) == 1:
        core["connectivity"]       = 0
        core["nearest_neighbor_m"] = np.nan
        core["connectivity_class"] = "faible"
    else:
        dist = distance_matrix(centroids, centroids)

        connected = (dist <= threshold_m) & (dist > 0)
        core["connectivity"] = connected.sum(axis=1)

        dist_no_diag = dist.copy()
        np.fill_diagonal(dist_no_diag, np.inf)
        core["nearest_neighbor_m"] = dist_no_diag.min(axis=1)

        q33 = np.quantile(core["connectivity"], 0.33)
        q66 = np.quantile(core["connectivity"], 0.66)
        core["connectivity_class"] = core["connectivity"].apply(
            lambda x: _classify_connectivity(float(x), q33, q66)
        )

    core["area_ha"] = core["area_m2"] / 1e4

    summary_df = (
        core.groupby("cover_label")
        .agg(
            nb_patches              =("patch_id",           "size"),
            connectivite_moyenne    =("connectivity",       "mean"),
            connectivite_mediane    =("connectivity",       "median"),
            connectivite_max        =("connectivity",       "max"),
            distance_voisin_moy     =("nearest_neighbor_m","mean"),
            surface_totale_ha       =("area_ha",            "sum"),
            surface_moy_ha          =("area_ha",            "mean"),
        )
        .reset_index()
        .sort_values("connectivite_moyenne", ascending=False)
        .reset_index(drop=True)
    )

    nn_vals = core["nearest_neighbor_m"]
    global_stats = {
        "threshold_m":                          float(threshold_m),
        "nb_patches":                           int(len(core)),
        "nb_habitats_coeur":                    int(core["cover_label"].nunique()),
        "connectivite_moyenne_globale":         float(core["connectivity"].mean()),
        "connectivite_mediane_globale":         float(core["connectivity"].median()),
        "distance_voisin_moy_globale":          float(nn_vals.mean()) if nn_vals.notna().any() else float("nan"),
        "part_patches_connectivite_faible_%":   float((core["connectivity_class"] == "faible").mean() * 100),
        "part_patches_connectivite_elevee_%":   float((core["connectivity_class"] == "élevée").mean() * 100),
    }

    return core, summary_df, global_stats


# ============================================================================
# COMMENTAIRE AUTOMATIQUE
# ============================================================================

def build_connectivity_comment(
    summary_df: pd.DataFrame,
    global_stats: Dict[str, float],
) -> str:
    """Génère un diagnostic textuel court à partir des métriques de connectivité."""
    if summary_df is None or summary_df.empty:
        return "Aucun résultat de connectivité disponible."

    mean_conn  = global_stats.get("connectivite_moyenne_globale", float("nan"))
    low_share  = global_stats.get("part_patches_connectivite_faible_%", float("nan"))
    high_share = global_stats.get("part_patches_connectivite_elevee_%", float("nan"))
    mean_nn    = global_stats.get("distance_voisin_moy_globale", float("nan"))
    threshold  = global_stats.get("threshold_m", float("nan"))

    parts: List[str] = []

    if mean_conn >= 30:
        parts.append(
            f"Connectivité **élevée** ({mean_conn:.1f} voisins en moyenne à moins de {threshold:.0f} m) — réseau écologique globalement fonctionnel."
        )
    elif mean_conn >= 10:
        parts.append(
            f"Connectivité **intermédiaire** ({mean_conn:.1f} voisins en moyenne) — continuités présentes mais inégalement distribuées."
        )
    else:
        parts.append(
            f"Connectivité **faible** ({mean_conn:.1f} voisins en moyenne) — réseau écologique morcelé ou partiellement rompu."
        )

    if pd.notna(low_share) and low_share >= 30:
        parts.append(f"**{low_share:.1f}%** des patches en connectivité faible → secteurs à reconnecter en priorité.")

    if pd.notna(high_share) and high_share >= 30:
        parts.append(f"**{high_share:.1f}%** des patches en connectivité élevée → noyaux de continuité à préserver.")

    if pd.notna(mean_nn):
        dist_status = "favorable" if mean_nn <= threshold else "au-dessus du seuil retenu"
        parts.append(f"Distance moyenne au plus proche voisin : **{mean_nn:.0f} m** ({dist_status}).")

    if not summary_df.empty:
        best  = summary_df.iloc[0]
        worst = summary_df.iloc[-1]
        parts.append(
            f"Habitat le mieux connecté : **{best['cover_label']}** ({best['connectivite_moyenne']:.1f}). "
            f"Moins connecté : **{worst['cover_label']}** ({worst['connectivite_moyenne']:.1f})."
        )

    return "\n\n".join(parts)
