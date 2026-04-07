"""
ecosys_sb.py — Score Biodiversité (SB)
========================================
Calcule un score de biodiversité par habitat à partir de :
  - observations GBIF (CSV)
  - occupation du sol (GeoJSON)

SB combine : richesse spécifique, densité, abondance,
fragmentation, Shannon et rareté (proxy fréquence ou IUCN).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================================
# PARAMÈTRES (modifiables)
# ============================================================================

GBIF_FILE      = "data/GBIF-lapeyruche.csv"
LANDCOVER_FILE = "data/occ-lapeyruche.geojson"
METRIC_CRS     = "EPSG:2154"

# Pondérations du SB — doivent sommer à 1.0
WEIGHTS_SB: Dict[str, float] = {
    "score_richesse":      0.25,
    "score_densite":       0.20,
    "score_abondance":     0.15,
    "score_fragmentation": 0.15,
    "score_shannon":       0.15,
    "score_rarity":        0.10,
}

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
# UTILITAIRES
# ============================================================================

def normalize_minmax(series: pd.Series) -> pd.Series:
    """Min-max vers [0, 1]. Renvoie 0 si série constante ou vide."""
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def compute_shannon_index(species_series: pd.Series) -> float:
    """H = -sum(p_i * ln(p_i)) sur les observations d'un habitat."""
    counts = species_series.value_counts()
    if counts.empty:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


# ============================================================================
# CHARGEMENT
# ============================================================================

def load_gbif_points(gbif_file: str) -> gpd.GeoDataFrame:
    """Charge le CSV GBIF → GeoDataFrame de points WGS84.
    Colonnes requises : title, latitude, longitude."""
    df = pd.read_csv(gbif_file)
    missing = {"title", "latitude", "longitude"} - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes GBIF manquantes : {missing}")
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    if "startdate" in df.columns:
        df["startdate"] = pd.to_datetime(df["startdate"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "title"])
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )


def load_landcover_polygons(landcover_file: str) -> gpd.GeoDataFrame:
    """Charge le GeoJSON et attribue l'habitat dominant à chaque polygone."""
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
    return occ


# ============================================================================
# CROISEMENT SPATIAL
# ============================================================================

def spatial_join(
    gbif_points: gpd.GeoDataFrame,
    landcover: gpd.GeoDataFrame,
    metric_crs: str = METRIC_CRS,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Jointure spatiale GBIF × occupation du sol en CRS métrique.
    Retourne (points reprojetés, polygones reprojetés + surfaces, jointure)."""
    pts_m = gbif_points.to_crs(metric_crs)
    occ_m = landcover.to_crs(metric_crs)
    occ_m["area_m2"] = occ_m.geometry.area
    joined = gpd.sjoin(
        pts_m,
        occ_m[["cover_label", "cover_code_max", "area_m2", "geometry"]],
        how="left", predicate="within",
    )
    return pts_m, occ_m, joined


# ============================================================================
# MÉTRIQUES PAR HABITAT
# ============================================================================

def compute_area_by_habitat(occ_m: gpd.GeoDataFrame) -> pd.DataFrame:
    """Surface totale (m², km², ha) et part surfacique (%) par habitat."""
    df    = occ_m.groupby("cover_label", dropna=False)["area_m2"].sum().reset_index(name="surface_m2")
    total = df["surface_m2"].sum()
    df["surface_km2"]    = df["surface_m2"] / 1e6
    df["surface_ha"]     = df["surface_m2"] / 1e4
    df["part_surface_%"] = df["surface_m2"] / total * 100
    return df


def compute_fragmentation(occ_m: gpd.GeoDataFrame) -> pd.DataFrame:
    """fragmentation_simple = nb polygones / km² — élevé = morcelé."""
    frag = (
        occ_m.groupby("cover_label")
        .agg(
            nb_polygones       =("geometry", "size"),
            surface_totale_m2  =("area_m2",  "sum"),
            surface_moyenne_m2 =("area_m2",  "mean"),
            surface_mediane_m2 =("area_m2",  "median"),
        )
        .reset_index()
    )
    frag["surface_totale_km2"]   = frag["surface_totale_m2"] / 1e6
    frag["fragmentation_simple"] = (
        frag["nb_polygones"] / frag["surface_totale_km2"].replace(0, np.nan)
    ).fillna(0)
    return frag


def compute_biodiversity_metrics(joined: gpd.GeoDataFrame, area_df: pd.DataFrame) -> pd.DataFrame:
    """Observations, espèces uniques et densités par km² par habitat."""
    data = joined.dropna(subset=["cover_label"]).copy()
    metrics = (
        data.groupby("cover_label")
        .agg(observations=("title", "size"), especes_uniques=("title", "nunique"))
        .reset_index()
    )
    metrics = metrics.merge(area_df[["cover_label", "surface_km2", "part_surface_%"]],
                            on="cover_label", how="left")
    metrics["obs_par_km2"]     = (metrics["observations"]    / metrics["surface_km2"].replace(0, np.nan)).fillna(0)
    metrics["especes_par_km2"] = (metrics["especes_uniques"] / metrics["surface_km2"].replace(0, np.nan)).fillna(0)
    return metrics


def compute_shannon_by_habitat(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    """Indice de Shannon H par habitat."""
    return (
        joined.dropna(subset=["cover_label"])
        .groupby("cover_label")["title"]
        .apply(compute_shannon_index)
        .reset_index(name="shannon")
    )


def compute_rarity_by_habitat(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    """Rareté proxy = moyenne(1 / fréquence_espèce) par habitat.
    ⚠ Proxy statistique — pas un statut de conservation officiel."""
    data = joined.dropna(subset=["cover_label"]).copy()
    freq = data["title"].value_counts()
    data["rarity_weight"] = data["title"].map(lambda x: 1 / freq[x])
    return data.groupby("cover_label")["rarity_weight"].mean().reset_index(name="rarity_score")


# ============================================================================
# SCORE SB
# ============================================================================

def compute_score_sb_by_habitat(
    biodiversity_df: pd.DataFrame,
    fragmentation_df: pd.DataFrame,
    shannon_df: pd.DataFrame,
    rarity_df: pd.DataFrame,
    weights: Dict[str, float] = WEIGHTS_SB,
) -> pd.DataFrame:
    """
    SB = combinaison linéaire pondérée de 6 composantes normalisées [0, 1].
    Fragmentation inversée : élevée = mauvais → score faible.
    """
    df = (
        biodiversity_df
        .merge(fragmentation_df[["cover_label", "fragmentation_simple"]], on="cover_label", how="left")
        .merge(shannon_df,  on="cover_label", how="left")
        .merge(rarity_df,   on="cover_label", how="left")
        .fillna(0)
    )
    df["score_richesse"]      = normalize_minmax(df["especes_uniques"])
    df["score_densite"]       = normalize_minmax(df["especes_par_km2"])
    df["score_abondance"]     = normalize_minmax(df["obs_par_km2"])
    df["score_shannon"]       = normalize_minmax(df["shannon"])
    df["score_rarity"]        = normalize_minmax(df["rarity_score"])
    df["score_fragmentation"] = 1 - normalize_minmax(df["fragmentation_simple"])

    df["score_sb"]     = sum(weights[k] * df[k] for k in weights)
    df["score_sb_100"] = df["score_sb"] * 100

    return df.sort_values("score_sb_100", ascending=False).reset_index(drop=True)


def compute_global_score_sb(score_sb_df: pd.DataFrame) -> float:
    """Score global SB = moyenne pondérée par surface (/100)."""
    df    = score_sb_df.copy()
    total = df["surface_km2"].sum()
    if total == 0:
        return 0.0
    df["_w"] = df["surface_km2"] / total
    return float((df["score_sb"] * df["_w"]).sum() * 100)


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def calculate_score_sb(
    gbif_file: str,
    landcover_file: str,
    metric_crs: str = METRIC_CRS,
) -> Tuple[pd.DataFrame, float]:
    """
    Pipeline SB complet.
    Retourne (DataFrame scores par habitat, score global SB /100).
    """
    gbif_pts  = load_gbif_points(gbif_file)
    landcover = load_landcover_polygons(landcover_file)
    _, occ_m, joined = spatial_join(gbif_pts, landcover, metric_crs)

    area_df   = compute_area_by_habitat(occ_m)
    bio_df    = compute_biodiversity_metrics(joined, area_df)
    frag_df   = compute_fragmentation(occ_m)
    shan_df   = compute_shannon_by_habitat(joined)
    rarity_df = compute_rarity_by_habitat(joined)

    scores       = compute_score_sb_by_habitat(bio_df, frag_df, shan_df, rarity_df)
    global_score = compute_global_score_sb(scores)
    return scores, global_score


# ============================================================================
# TEST LOCAL
# ============================================================================

if __name__ == "__main__":
    for f in [GBIF_FILE, LANDCOVER_FILE]:
        if not Path(f).exists():
            raise FileNotFoundError(f"Fichier introuvable : {f}")

    scores, global_sb = calculate_score_sb(GBIF_FILE, LANDCOVER_FILE)
    print(f"\n{'='*60}\nSCORE GLOBAL SB : {global_sb:.2f} / 100\n{'='*60}")
    print(scores[["cover_label", "score_sb_100", "surface_km2",
                  "especes_uniques", "obs_par_km2", "shannon"]].to_string(index=False))
