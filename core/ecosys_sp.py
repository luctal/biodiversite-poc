"""
ecosys_sp.py — Score Paysage (SP)
===================================
Calcule un score de qualité paysagère à partir de l'occupation du sol uniquement.
Aucune donnée GBIF requise.

SP = 100 × naturalité = 100 × Σ(pᵢ × wᵢ)
  pᵢ = proportion surfacique de la classe i
  wᵢ = poids écologique de la classe i

Indicateurs complémentaires (descriptifs, non intégrés au score) :
  Shannon paysage, Piélou, richesse en classes.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================================
# PARAMÈTRES (modifiables)
# ============================================================================

LANDCOVER_FILE = "data/occ-lapeyruche.geojson"
METRIC_CRS     = "EPSG:2154"

# Poids écologiques par habitat
# 1.0 = très naturel / très favorable à la biodiversité
# 0.0 = totalement artificialisé
HABITAT_WEIGHTS: Dict[str, float] = {
    # Urbain
    "Urbain dense":                  0.00,
    "Tissu urbain continu":          0.00,
    "Zone industrielle/commerciale": 0.05,
    "Urbain diffus":                 0.15,
    "Tissu urbain discontinu":       0.15,
    "Routes":                        0.00,
    "Surface minérale":              0.10,
    "Serres":                        0.05,
    # Agriculture intensive
    "Maïs":                          0.20,
    "Tubercules/racines":            0.20,
    "Céréales/pailles":              0.25,
    "Colza":                         0.25,
    "Soja":                          0.25,
    "Oléagineux d'hiver":            0.25,
    "Protéagineux":                  0.30,
    "Tournesol":                     0.30,
    "Riz":                           0.30,
    # Agriculture extensive
    "Vignes":                        0.45,
    "Vergers":                       0.50,
    "Prairie":                       0.70,
    # Milieux semi-naturels
    "Pelouse":                       0.80,
    "Landes":                        0.80,
    "Plage/dune":                    0.75,
    "Glace/neige":                   0.85,
    # Milieux naturels
    "Eau":                           0.85,
    "Conifères":                     0.85,
    "Feuillus":                      1.00,
}

# Poids par défaut pour les classes absentes du dictionnaire
DEFAULT_HABITAT_WEIGHT = 0.40

# Codes thématiques du GeoJSON → polygone hérite de la colonne à valeur max
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


# ============================================================================
# CHARGEMENT
# ============================================================================

def load_landcover(landcover_file: str, metric_crs: str = METRIC_CRS) -> gpd.GeoDataFrame:
    """Charge le GeoJSON, attribue l'habitat dominant, calcule les surfaces."""
    occ = gpd.read_file(landcover_file)
    missing = [c for c in THEME_COLS if c not in occ.columns]
    if missing:
        raise ValueError(f"Colonnes thématiques manquantes : {missing}")
    if occ.crs is None:
        warnings.warn("CRS absent — EPSG:4326 supposé.")
        occ = occ.set_crs("EPSG:4326")
    for col in THEME_COLS:
        occ[col] = pd.to_numeric(occ[col], errors="coerce").fillna(0)
    occ["cover_code_max"] = occ[THEME_COLS].idxmax(axis=1)
    occ["cover_label"]    = occ["cover_code_max"].map(LABEL_FR).fillna(occ["cover_code_max"])

    occ_m = occ.to_crs(metric_crs)
    occ_m["area_m2"] = occ_m.geometry.area
    return occ_m


# ============================================================================
# INDICATEURS PAYSAGE
# ============================================================================

def compute_class_areas(occ_m: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Agrège les surfaces par classe, calcule proportion et poids écologique.
    Retourne un DataFrame par classe trié par surface décroissante.
    """
    df = (
        occ_m.groupby("cover_label", dropna=False)["area_m2"]
        .sum()
        .reset_index(name="surface_m2")
    )
    total = df["surface_m2"].sum()
    df["surface_km2"]      = df["surface_m2"] / 1e6
    df["surface_ha"]       = df["surface_m2"] / 1e4
    df["part_surface_%"]   = df["surface_m2"] / total * 100
    df["proportion"]       = df["surface_m2"] / total
    df["poids_ecologique"] = df["cover_label"].map(HABITAT_WEIGHTS).fillna(DEFAULT_HABITAT_WEIGHT)
    df["contribution_sp"]  = df["proportion"] * df["poids_ecologique"]
    df["poids_connu"]      = df["cover_label"].isin(HABITAT_WEIGHTS)

    return df.sort_values("surface_m2", ascending=False).reset_index(drop=True)


def compute_shannon_paysage(areas_df: pd.DataFrame) -> float:
    """Shannon paysage H = -Σ pᵢ ln(pᵢ) sur les classes d'occupation du sol."""
    p = areas_df["proportion"].values
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def compute_pielou(shannon: float, n_classes: int) -> float:
    """Équitabilité de Piélou J = H / ln(S). 1 = équilibré, 0 = dominance."""
    if n_classes <= 1:
        return 0.0
    return shannon / math.log(n_classes)


def compute_naturality(areas_df: pd.DataFrame) -> float:
    """Naturalité = Σ(proportion × poids_écologique) — entre 0 et 1."""
    return float(areas_df["contribution_sp"].sum())


def compute_fragmentation_landscape(occ_m: gpd.GeoDataFrame) -> pd.DataFrame:
    """fragmentation_simple = nb polygones / km² par classe."""
    frag = (
        occ_m.groupby("cover_label")
        .agg(
            nb_polygones       =("geometry", "size"),
            surface_totale_m2  =("area_m2",  "sum"),
            surface_moyenne_ha =("area_m2",  lambda x: x.mean() / 1e4),
            surface_mediane_ha =("area_m2",  lambda x: x.median() / 1e4),
        )
        .reset_index()
    )
    frag["surface_totale_km2"]   = frag["surface_totale_m2"] / 1e6
    frag["fragmentation_simple"] = (
        frag["nb_polygones"] / frag["surface_totale_km2"].replace(0, np.nan)
    ).fillna(0)
    return frag


# ============================================================================
# SCORE SP
# ============================================================================

def compute_score_sp(naturality: float) -> float:
    """SP = 100 × naturalité. Entre 0 et 100."""
    return round(100.0 * naturality, 4)


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def calculate_score_sp(
    landcover_file: str,
    metric_crs: str = METRIC_CRS,
) -> Dict[str, Any]:
    """
    Pipeline SP complet — ne nécessite aucune donnée GBIF.

    Retourne un dictionnaire avec :
      score_sp          : score paysage global (0–100)
      naturality        : naturalité brute (0–1)
      shannon_paysage   : diversité des classes
      pielou            : équitabilité
      richesse_classes  : nombre de classes distinctes
      surface_totale_ha : surface totale analysée
      classes_df        : DataFrame détaillé par classe
      classes_inconnues : classes sans poids explicite dans HABITAT_WEIGHTS
    """
    occ_m = load_landcover(landcover_file, metric_crs)

    areas_df  = compute_class_areas(occ_m)
    frag_df   = compute_fragmentation_landscape(occ_m)

    # Fusion fragmentation dans areas_df
    areas_df  = areas_df.merge(
        frag_df[["cover_label", "nb_polygones", "fragmentation_simple",
                 "surface_moyenne_ha", "surface_mediane_ha"]],
        on="cover_label", how="left",
    )

    naturality   = compute_naturality(areas_df)
    n_classes    = len(areas_df)
    shannon_pay  = compute_shannon_paysage(areas_df)
    pielou       = compute_pielou(shannon_pay, n_classes)
    score_sp     = compute_score_sp(naturality)
    total_ha     = float(areas_df["surface_ha"].sum())
    classes_unk  = areas_df.loc[~areas_df["poids_connu"], "cover_label"].tolist()

    return {
        "score_sp":           score_sp,
        "naturality":         round(naturality, 4),
        "shannon_paysage":    round(shannon_pay, 4),
        "pielou":             round(pielou, 4),
        "richesse_classes":   n_classes,
        "surface_totale_ha":  round(total_ha, 2),
        "classes_df":         areas_df,
        "classes_inconnues":  classes_unk,
    }


# ============================================================================
# TEST LOCAL
# ============================================================================

if __name__ == "__main__":
    if not Path(LANDCOVER_FILE).exists():
        raise FileNotFoundError(f"GeoJSON introuvable : {LANDCOVER_FILE}")

    result = calculate_score_sp(LANDCOVER_FILE)

    print(f"\n{'='*60}")
    print(f"SCORE SP (paysage) : {result['score_sp']:.2f} / 100")
    print(f"  Naturalité       : {result['naturality']:.4f}")
    print(f"  Shannon paysage  : {result['shannon_paysage']:.4f}")
    print(f"  Piélou           : {result['pielou']:.4f}")
    print(f"  Richesse classes : {result['richesse_classes']}")
    print(f"  Surface totale   : {result['surface_totale_ha']:.1f} ha")
    print(f"{'='*60}")

    if result["classes_inconnues"]:
        print(f"\n⚠ Classes sans poids (DEFAULT={DEFAULT_HABITAT_WEIGHT}) :")
        for c in result["classes_inconnues"]:
            print(f"  - {c}")

    print("\nTop 10 classes :")
    top = result["classes_df"].head(10)[
        ["cover_label", "surface_ha", "part_surface_%",
         "poids_ecologique", "contribution_sp", "fragmentation_simple"]
    ]
    print(top.round(3).to_string(index=False))
