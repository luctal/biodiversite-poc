"""
score_v3.py — Score synthétique V3
=====================================
Fusionne trois scores indépendants :
  SB  — biodiversité GBIF       (ecosys_sb.py)
  SP  — paysage / occupation sol (ecosys_sp.py)
  SC  — connectivité spatiale    (connec.py)

Formule :
  V3 = W_SB × SB + W_SP × SP + W_SC × SC

Les pondérations sont modifiables en tête de fichier.
Si un score est absent (données manquantes), sa pondération
est redistribuée proportionnellement entre les scores disponibles.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# PARAMÈTRES (modifiables)
# ============================================================================

# Pondérations du V3 — doivent sommer à 1.0
WEIGHTS_V3: Dict[str, float] = {
    "sb":   0.40,   # Score biodiversité GBIF
    "sp":   0.35,   # Score paysage (naturalité)
    "sc":   0.25,   # Score connectivité spatiale
}

# Pondérations internes du score de connectivité SC
WEIGHTS_SC: Dict[str, float] = {
    "conn_mean":     0.60,   # Rang connectivité moyenne
    "conn_distance": 0.30,   # Inverse distance voisin
    "conn_patches":  0.10,   # Rang nombre de patches
}


# ============================================================================
# UTILITAIRES DE NORMALISATION
# ============================================================================

def percentile_rank(series: pd.Series) -> pd.Series:
    """Rang percentile [0, 1] — valeur élevée → score élevé."""
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    if s.empty or s.nunique() <= 1:
        return pd.Series(0.0, index=s.index)
    return s.rank(method="average", pct=True)


def normalize_inverse(series: pd.Series) -> pd.Series:
    """Distance courte → score proche de 1, distance grande → proche de 0."""
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return pd.Series(0.0, index=s.index)
    s = s.fillna(s.max())
    if s.max() == s.min():
        return pd.Series(0.0, index=s.index)
    return 1 - (s - s.min()) / (s.max() - s.min())


def redistribute_weights(weights: Dict[str, float], missing_keys: list) -> Dict[str, float]:
    """
    Redistribue proportionnellement les pondérations des scores manquants
    vers les scores disponibles.
    """
    available = {k: v for k, v in weights.items() if k not in missing_keys}
    if not available:
        return weights
    total_available = sum(available.values())
    return {k: v / total_available for k, v in available.items()}


# ============================================================================
# SCORE DE CONNECTIVITÉ (SC) PAR HABITAT
# ============================================================================

def compute_score_sc(conn_summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule un score de connectivité normalisé SC [0, 1] par habitat cœur.

    Composantes :
      60% → rang(connectivité_moyenne)
      30% → inverse normalisée(distance_voisin_moy)
      10% → rang(nb_patches)

    Colonnes requises : cover_label, connectivite_moyenne, distance_voisin_moy, nb_patches
    """
    required = {"cover_label", "connectivite_moyenne", "distance_voisin_moy", "nb_patches"}
    missing  = required - set(conn_summary_df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes (connectivité) : {missing}")

    df = conn_summary_df.copy()
    df["score_conn_mean"]     = percentile_rank(df["connectivite_moyenne"])
    df["score_conn_distance"] = normalize_inverse(df["distance_voisin_moy"])
    df["score_conn_patches"]  = percentile_rank(df["nb_patches"])

    w = WEIGHTS_SC
    df["score_sc"]     = (
        w["conn_mean"]     * df["score_conn_mean"]     +
        w["conn_distance"] * df["score_conn_distance"] +
        w["conn_patches"]  * df["score_conn_patches"]
    )
    df["score_sc_100"] = df["score_sc"] * 100

    return df[[
        "cover_label",
        "connectivite_moyenne", "distance_voisin_moy", "nb_patches",
        "score_conn_mean", "score_conn_distance", "score_conn_patches",
        "score_sc", "score_sc_100",
    ]].copy()


# ============================================================================
# SCORE V3 PAR HABITAT
# ============================================================================

def compute_score_v3_by_habitat(
    sb_df: Optional[pd.DataFrame],
    sp_result: Optional[dict],
    sc_df: Optional[pd.DataFrame],
    weights: Dict[str, float] = WEIGHTS_V3,
) -> pd.DataFrame:
    """
    Fusionne SB, SP et SC en un score V3 par habitat.

    Paramètres
    ----------
    sb_df      : DataFrame scores SB par habitat (depuis ecosys_sb.py)
                 Colonnes requises : cover_label, score_sb, score_sb_100, surface_km2
    sp_result  : dict résultat de calculate_score_sp() (depuis ecosys_sp.py)
                 Doit contenir 'classes_df' avec cover_label, proportion, poids_ecologique
    sc_df      : DataFrame scores SC par habitat (depuis compute_score_sc())
                 Colonnes requises : cover_label, score_sc, score_sc_100
    weights    : pondérations V3 (sb, sp, sc)

    Logique de fusion
    -----------------
    - La base des habitats est celle du SB si disponible, sinon celle du SP.
    - SP est un score global, pas par habitat. Pour la fusion par habitat,
      on utilise la contribution SP de chaque classe (proportion × poids).
      Cela permet de voir quel habitat « tire » le SP vers le haut ou le bas.
    - SC est disponible uniquement pour les habitats cœur (Feuillus, Prairie, Eau…).
      Les habitats absents du SC reçoivent score_sc = 0.
    - Si SB est absent (pas de GBIF), V3 = SP pondéré + SC pondéré, avec
      redistribution automatique des poids.

    Retour
    ------
    DataFrame trié par score_v3_100 décroissant, avec colonnes :
      cover_label, surface_km2, score_sb_100, score_sp_100, score_sc_100,
      score_v3, score_v3_100, + détails composantes
    """
    # Déterminer quels scores sont disponibles
    has_sb = sb_df is not None and not sb_df.empty and "score_sb" in sb_df.columns
    has_sp = sp_result is not None and "classes_df" in sp_result
    has_sc = sc_df is not None and not sc_df.empty and "score_sc" in sc_df.columns

    if not has_sb and not has_sp:
        raise ValueError("Au moins SB ou SP doit être disponible pour calculer V3.")

    # Redistribution des poids si un score est absent
    missing_keys = ([k for k in ["sb", "sp", "sc"]
                     if (k == "sb" and not has_sb)
                     or (k == "sp" and not has_sp)
                     or (k == "sc" and not has_sc)])
    eff_weights = redistribute_weights(weights, missing_keys)

    # Base habitats : SB en priorité, SP sinon
    if has_sb:
        df = sb_df[["cover_label", "surface_km2", "part_surface_%",
                    "score_sb", "score_sb_100"]].copy()
    else:
        sp_classes = sp_result["classes_df"]
        df = sp_classes[["cover_label", "surface_ha", "part_surface_%"]].copy()
        df["surface_km2"] = df["surface_ha"] / 100
        df = df.drop(columns=["surface_ha"])
        df["score_sb"]     = 0.0
        df["score_sb_100"] = 0.0

    # Intégration SP par habitat (contribution normalisée)
    if has_sp:
        sp_classes = sp_result["classes_df"][
            ["cover_label", "contribution_sp", "poids_ecologique", "proportion"]
        ].copy()
        # Normalisation : contribution_sp → score [0, 1] relatif au max observé
        max_contrib = sp_classes["contribution_sp"].max()
        sp_classes["score_sp"] = (
            sp_classes["contribution_sp"] / max_contrib if max_contrib > 0
            else 0.0
        )
        sp_classes["score_sp_100"] = sp_classes["score_sp"] * 100
        df = df.merge(
            sp_classes[["cover_label", "score_sp", "score_sp_100",
                        "poids_ecologique", "contribution_sp"]],
            on="cover_label", how="left",
        )
    else:
        df["score_sp"]       = 0.0
        df["score_sp_100"]   = 0.0
        df["poids_ecologique"] = 0.0
        df["contribution_sp"]  = 0.0

    # Intégration SC par habitat
    if has_sc:
        df = df.merge(
            sc_df[["cover_label", "score_sc", "score_sc_100"]],
            on="cover_label", how="left",
        )
    else:
        df["score_sc"]     = 0.0
        df["score_sc_100"] = 0.0

    df["score_sc"]     = df["score_sc"].fillna(0.0)
    df["score_sc_100"] = df["score_sc_100"].fillna(0.0)
    df["score_sp"]     = df["score_sp"].fillna(0.0)
    df["score_sp_100"] = df["score_sp_100"].fillna(0.0)

    # Score V3
    df["score_v3"] = (
        eff_weights.get("sb", 0) * df["score_sb"] +
        eff_weights.get("sp", 0) * df["score_sp"] +
        eff_weights.get("sc", 0) * df["score_sc"]
    )
    df["score_v3_100"] = df["score_v3"] * 100

    # Méta : pondérations effectives utilisées
    df["_w_sb"] = eff_weights.get("sb", 0)
    df["_w_sp"] = eff_weights.get("sp", 0)
    df["_w_sc"] = eff_weights.get("sc", 0)

    return df.sort_values("score_v3_100", ascending=False).reset_index(drop=True)


# ============================================================================
# SCORES GLOBAUX
# ============================================================================

def compute_global_score_sb(sb_df: pd.DataFrame) -> float:
    """Score global SB = moyenne pondérée par surface (/100)."""
    total = sb_df["surface_km2"].sum()
    if total == 0: return 0.0
    return float((sb_df["score_sb"] * sb_df["surface_km2"] / total).sum() * 100)


def compute_global_score_sp(sp_result: dict) -> float:
    """Score global SP (directement disponible dans le dict résultat)."""
    return float(sp_result.get("score_sp", 0.0))


def compute_global_score_sc(sc_df: pd.DataFrame, conn_summary_df: pd.DataFrame) -> float:
    """Score global SC = moyenne pondérée par surface des patches."""
    df = sc_df.merge(
        conn_summary_df[["cover_label", "surface_totale_ha"]],
        on="cover_label", how="left",
    )
    total = df["surface_totale_ha"].sum()
    if total == 0: return 0.0
    return float((df["score_sc"] * df["surface_totale_ha"] / total).sum() * 100)


def compute_global_score_v3(v3_df: pd.DataFrame) -> float:
    """Score global V3 = moyenne pondérée par surface des habitats (/100)."""
    total = v3_df["surface_km2"].sum()
    if total == 0: return 0.0
    return float((v3_df["score_v3"] * v3_df["surface_km2"] / total).sum() * 100)


# ============================================================================
# RÉSUMÉ DES PONDÉRATIONS EFFECTIVES
# ============================================================================

def get_effective_weights(v3_df: pd.DataFrame) -> Dict[str, float]:
    """Retourne les pondérations effectives utilisées dans le V3."""
    if v3_df.empty:
        return {}
    row = v3_df.iloc[0]
    return {
        "sb": float(row.get("_w_sb", 0)),
        "sp": float(row.get("_w_sp", 0)),
        "sc": float(row.get("_w_sc", 0)),
    }
