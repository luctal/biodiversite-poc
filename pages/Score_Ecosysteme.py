"""
visu.py — Interface Streamlit — Score Écosystème V3
=====================================================
Trois scores indépendants + synthèse V3 :
  SB  — Biodiversité GBIF       (ecosys_sb.py)  — optionnel
  SP  — Paysage / occupation sol (ecosys_sp.py)  — requis
  SC  — Connectivité spatiale    (connec.py)     — requis
  V3  — Synthèse SB + SP + SC   (score_v3.py)
"""

from __future__ import annotations

import json
import tempfile
import traceback
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from core.connec import DEFAULT_CORE_HABITATS, build_connectivity_comment, calculate_connectivity
from core.ecosys_sb import calculate_score_sb
from core.ecosys_sp import calculate_score_sp
from core.score_v3 import (
    compute_score_sc,
    compute_score_v3_by_habitat,
    compute_global_score_sb,
    compute_global_score_sp,
    compute_global_score_sc,
    compute_global_score_v3,
    get_effective_weights,
    WEIGHTS_V3,
)

# ============================================================================
# CHARTE GRAPHIQUE
# ============================================================================

C_NAVY        = "#2571A3"
C_NAVY_LIGHT  = "#CFE8F9"
C_GREEN_DARK  = "#2D4E28"
C_GREEN_LIGHT = "#A2CB86"
C_BEIGE       = "#FBF4EC"
C_OLIVE       = "#C1B900"
C_BURGUNDY    = "#86193F"
C_VIOLET      = "#4F479B"
C_GREY        = "#D3D3D3"
C_TEXT        = "#2A2A2A"

C_SB = C_NAVY
C_SP = C_GREEN_DARK
C_SC = C_VIOLET
C_V3 = C_OLIVE


def score_color(val: float) -> str:
    if val >= 75: return C_GREEN_DARK
    if val >= 60: return C_GREEN_LIGHT
    if val >= 45: return C_OLIVE
    return C_BURGUNDY


def score_label(val: float) -> str:
    if val >= 75: return "Tres favorable"
    if val >= 60: return "Favorable"
    if val >= 45: return "Intermediaire"
    return "Defavorable"


# ============================================================================
# CONFIG PAGE & CSS
# ============================================================================

st.set_page_config(page_title="Score Ecosysteme V3", page_icon=None,
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
.stApp {{ background-color: {C_BEIGE}; }}
div[data-testid="stSidebar"] {{ background-color: #f0ece4; border-right: 1px solid #e0d8cc; }}
.stTabs [data-baseweb="tab-list"] {{
    background-color: {C_NAVY_LIGHT}; border-radius: 6px 6px 0 0; padding: 2px 4px 0 4px; gap: 2px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent; color: {C_NAVY}; font-weight: 600;
    font-size: 0.82rem; letter-spacing: 0.04em; padding: 0.45rem 1.1rem;
    border-radius: 4px 4px 0 0; border: none;
}}
.stTabs [aria-selected="true"] {{
    background-color: {C_BEIGE} !important; color: {C_GREEN_DARK} !important;
    border-bottom: 2px solid {C_GREEN_DARK};
}}
[data-testid="stMetricValue"] {{
    font-size: 1.9rem; font-weight: 700; color: {C_TEXT}; font-family: "Georgia", serif;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.72rem; color: #666; text-transform: uppercase; letter-spacing: 0.06em;
}}
[data-testid="metric-container"] {{
    background-color: white; border: 1px solid #e8e0d4; border-radius: 6px; padding: 0.8rem 1rem;
}}
h1, h2, h3 {{ font-family: "Georgia", serif; color: {C_TEXT}; }}
hr {{ border-color: #ddd5c8; }}
div[data-testid="stButton"] > button[kind="primary"] {{
    background-color: {C_NAVY}; color: white; border: none; font-weight: 600;
}}
div[data-testid="stButton"] > button[kind="primary"]:hover {{ background-color: {C_GREEN_DARK}; }}
div[data-testid="stDownloadButton"] > button {{
    background-color: transparent; border: 1px solid {C_NAVY}; color: {C_NAVY}; font-size: 0.78rem;
}}
[data-testid="stDataFrame"] {{ border: 1px solid #e0d8cc; border-radius: 4px; }}
[data-testid="stCaptionContainer"] {{ color: #555; font-size: 0.82rem; line-height: 1.5; }}
div[data-testid="stSidebar"] label {{
    font-size: 0.8rem; font-weight: 600; color: {C_TEXT}; text-transform: uppercase;
}}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPERS
# ============================================================================

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Calibri, sans-serif", color=C_TEXT, size=11),
    title_font=dict(size=12, color=C_NAVY, family="Calibri, sans-serif"),
)


def section_header(text: str):
    st.markdown(
        f"<p style='font-size:0.75rem;font-weight:700;color:{C_NAVY};text-transform:uppercase;"
        f"letter-spacing:0.07em;margin-bottom:0.3rem;border-left:3px solid {C_NAVY};"
        f"padding-left:0.5rem;'>{text}</p>", unsafe_allow_html=True)


def status_badge(val: float) -> str:
    c, l = score_color(val), score_label(val)
    return (f"<span style='background:{c};color:white;font-size:0.7rem;font-weight:700;"
            f"padding:2px 8px;border-radius:3px;letter-spacing:0.05em;'>{l}</span>")


def save_file(f, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(f.getbuffer())
        return tmp.name


# ============================================================================
# GRAPHIQUES
# ============================================================================

def gauge(value: float, title: str, color: str | None = None) -> go.Figure:
    c = color or score_color(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": " /100", "font": {"size": 30, "color": c, "family": "Georgia, serif"}},
        title={"text": title.upper(), "font": {"size": 10, "color": C_NAVY, "family": "Calibri"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": C_GREY, "tickfont": {"size": 9}},
            "bar": {"color": c, "thickness": 0.22},
            "bgcolor": "#f8f4ef", "borderwidth": 0,
            "steps": [
                {"range": [0, 45], "color": "#f5e6e8"}, {"range": [45, 60], "color": "#faf4dc"},
                {"range": [60, 75], "color": "#e8f0e2"}, {"range": [75, 100], "color": "#d4e8cc"},
            ],
            "threshold": {"line": {"color": c, "width": 2}, "value": value},
        },
    ))
    fig.update_layout(height=195, margin=dict(t=40, b=5, l=15, r=15), **PLOTLY_BASE)
    return fig


def hbar(df, x, y, title, color_col=None, color_fixed=C_NAVY):
    dp = df.sort_values(x, ascending=True)
    colors = [score_color(v) for v in dp[color_col]] if color_col else color_fixed
    fig = go.Figure(go.Bar(
        x=dp[x], y=dp[y], orientation="h",
        marker_color=colors, marker_line_width=0,
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=max(220, 38 * len(dp)),
        margin=dict(t=36, b=10, l=10, r=10),
        xaxis=dict(showgrid=True, gridcolor="#e8e0d4", zeroline=False),
        yaxis=dict(showgrid=False), **PLOTLY_BASE,
    )
    return fig


def radar_sb(scores_df):
    cols   = ["score_richesse", "score_densite", "score_abondance",
              "score_fragmentation", "score_shannon", "score_rarity"]
    labels = ["Richesse", "Densite", "Abondance", "Fragmentation", "Shannon", "Rarete"]
    avail  = [c for c in cols if c in scores_df.columns]
    if not avail:
        return None
    palette = [C_NAVY, C_GREEN_DARK, C_VIOLET, C_OLIVE, C_BURGUNDY, C_GREEN_LIGHT]
    fig = go.Figure()
    for i, (_, row) in enumerate(scores_df.iterrows()):
        vals = [float(row.get(c, 0)) for c in avail]
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=labels[:len(avail)] + [labels[0]],
            name=row["cover_label"], fill="toself", opacity=0.35,
            line=dict(color=color, width=1.5), fillcolor=color,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=8), gridcolor="#ddd"),
            angularaxis=dict(tickfont=dict(size=9, color=C_TEXT)),
            bgcolor="#f8f4ef",
        ),
        height=400, margin=dict(t=40, b=20, l=40, r=140),
        legend=dict(orientation="v", x=1.05, font=dict(size=9)),
        title="Profil des composantes SB par habitat", **PLOTLY_BASE,
    )
    return fig


def monthly_histogram(data_df):
    if "startdate" not in data_df.columns:
        return None
    dates = pd.to_datetime(data_df["startdate"], errors="coerce").dropna()
    if dates.empty:
        return None
    monthly = dates.dt.to_period("M").value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=[str(p) for p in monthly.index], y=monthly.values,
        marker_color=C_NAVY, marker_line_width=0,
        hovertemplate="%{x} : %{y} obs.<extra></extra>",
    ))
    fig.update_layout(
        title="Dynamique temporelle — observations par mois",
        height=240, bargap=0.05,
        xaxis=dict(showgrid=False, tickangle=-45, tickfont=dict(size=8), title=""),
        yaxis=dict(showgrid=True, gridcolor="#e8e0d4",
                   title=dict(text="Observations", font=dict(size=9))),
        **PLOTLY_BASE,
    )
    return fig


def v3_comparison(v3_df, has_sb):
    df = v3_df.sort_values("score_v3_100", ascending=True)
    fig = go.Figure()
    if has_sb and "score_sb_100" in df.columns:
        fig.add_trace(go.Bar(y=df["cover_label"], x=df["score_sb_100"], name="SB",
                              orientation="h", marker_color=C_SB, marker_line_width=0, opacity=0.75))
    if "score_sp_100" in df.columns:
        fig.add_trace(go.Bar(y=df["cover_label"], x=df["score_sp_100"], name="SP",
                              orientation="h", marker_color=C_SP, marker_line_width=0, opacity=0.75))
    if "score_sc_100" in df.columns:
        fig.add_trace(go.Bar(y=df["cover_label"], x=df["score_sc_100"], name="SC",
                              orientation="h", marker_color=C_SC, marker_line_width=0, opacity=0.75))
    fig.add_trace(go.Bar(y=df["cover_label"], x=df["score_v3_100"], name="V3",
                          orientation="h", marker_color=C_V3, marker_line_width=0, opacity=1.0))
    fig.update_layout(
        barmode="group", height=max(280, 50 * len(df)),
        margin=dict(t=36, b=10, l=10, r=10),
        xaxis=dict(title="Score /100", showgrid=True, gridcolor="#e8e0d4", range=[0, 110]),
        yaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=9)),
        title="Comparaison SB / SP / SC / V3 par habitat", **PLOTLY_BASE,
    )
    return fig


# ============================================================================
# CARTES
# ============================================================================

_HABITAT_PALETTE = [
    [45, 78, 40], [37, 113, 163], [79, 71, 155], [134, 25, 63],
    [162, 203, 134], [193, 185, 0], [207, 232, 249], [93, 118, 83],
    [211, 211, 211], [251, 244, 236],
]


def _habitat_colors(gdf):
    col = next((c for c in ["cover_label", "habitat_label", "Classe"] if c in gdf.columns), None)
    if col is None:
        return [[100, 160, 100, 180]] * len(gdf)
    labels = gdf[col].unique().tolist()
    lut = {l: _HABITAT_PALETTE[i % len(_HABITAT_PALETTE)] for i, l in enumerate(labels)}
    return [lut.get(r, [150, 150, 150]) + [180] for r in gdf[col]]


def map_habitat(landcover_path):
    gdf = gpd.read_file(landcover_path)
    if gdf.empty: st.warning("GeoJSON vide."); return
    gdf = gdf.to_crs("EPSG:4326") if gdf.crs else gdf.set_crs("EPSG:4326")
    try:
        from core.connec import _ensure_cover_label
        gdf = _ensure_cover_label(gdf)
    except Exception:
        pass
    label_col = next((c for c in ["cover_label", "Classe", "name"] if c in gdf.columns), None)
    keep = ([label_col] if label_col else []) + ["geometry"]
    gdf_map = gdf[keep].copy()
    if label_col and label_col != "habitat_label":
        gdf_map = gdf_map.rename(columns={label_col: "habitat_label"})
    gdf_map["_color"] = _habitat_colors(gdf_map)
    bounds = gdf.total_bounds
    span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    zoom = 14 if span < 0.02 else 12 if span < 0.1 else 10
    c = gdf.union_all().centroid
    geojson = json.loads(gdf_map.drop(columns=["_color"]).to_json())
    colors = gdf_map["_color"].tolist()
    for i, feat in enumerate(geojson["features"]):
        feat["properties"]["_fill"] = colors[i]
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("GeoJsonLayer", geojson, opacity=1.0, stroked=True, filled=True,
                           get_line_color=[40, 40, 40], get_fill_color="properties._fill",
                           line_width_min_pixels=1, pickable=True, auto_highlight=True,
                           highlight_color=[255, 255, 255, 80])],
        initial_view_state=pdk.ViewState(latitude=c.y, longitude=c.x, zoom=zoom, pitch=0),
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"html": "<b>{habitat_label}</b>",
                 "style": {"backgroundColor": C_NAVY, "color": "white",
                            "fontSize": "12px", "padding": "6px 10px"}},
    ), height=460)


def map_gbif(gbif_path, max_pts=5000):
    df = pd.read_csv(gbif_path)
    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    if df.empty: st.warning("Aucun point GBIF."); return
    if len(df) > max_pts: df = df.sample(max_pts, random_state=42)
    cols = [c for c in ["title", "startdate", "latitude", "longitude"] if c in df.columns]
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("ScatterplotLayer", df[cols], get_position="[longitude, latitude]",
                           get_radius=18, get_fill_color=[37, 113, 163, 160], pickable=True)],
        initial_view_state=pdk.ViewState(latitude=df["latitude"].mean(),
                                          longitude=df["longitude"].mean(), zoom=12),
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"html": "<b>{title}</b><br/>{startdate}",
                 "style": {"backgroundColor": C_NAVY, "color": "white", "fontSize": "11px"}},
    ), height=420)


def map_connectivity(patches_gdf):
    if patches_gdf is None or patches_gdf.empty: st.warning("Aucun patch."); return
    gdf = patches_gdf.to_crs("EPSG:4326").copy()
    gdf["lon"] = gdf.geometry.centroid.x
    gdf["lat"] = gdf.geometry.centroid.y

    def cc(cls):
        return ([45, 78, 40, 200] if cls == "elevee" or cls == "élevée"
                else [193, 185, 0, 200] if "interm" in str(cls).lower()
                else [134, 25, 63, 200])

    gdf["_color"] = gdf["connectivity_class"].apply(cc)
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer(
            "ScatterplotLayer",
            gdf[["patch_id", "cover_label", "connectivity", "connectivity_class", "lon", "lat", "_color"]],
            get_position="[lon, lat]", get_radius=55, get_fill_color="_color", pickable=True,
        )],
        initial_view_state=pdk.ViewState(latitude=gdf["lat"].mean(),
                                          longitude=gdf["lon"].mean(), zoom=11),
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"html": "<b>{cover_label}</b><br/>Connectivite : {connectivity} — {connectivity_class}",
                 "style": {"backgroundColor": "#1a1a2e", "color": "white", "fontSize": "11px"}},
    ), height=420)
    st.markdown(
        f"<div style='display:flex;gap:1.5rem;font-size:0.75rem;color:{C_TEXT};margin-top:0.3rem;'>"
        f"<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
        f"background:{C_GREEN_DARK};margin-right:4px;'></span>Elevee</span>"
        f"<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
        f"background:{C_OLIVE};margin-right:4px;'></span>Intermediaire</span>"
        f"<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
        f"background:{C_BURGUNDY};margin-right:4px;'></span>Faible</span>"
        f"</div>", unsafe_allow_html=True,
    )


# ============================================================================
# CACHE
# ============================================================================

@st.cache_data(show_spinner=False)
def cached_score_sb(gbif_file, landcover_file, metric_crs):
    return calculate_score_sb(gbif_file, landcover_file, metric_crs)


@st.cache_data(show_spinner=False)
def cached_score_sp(landcover_file, metric_crs):
    return calculate_score_sp(landcover_file, metric_crs)


@st.cache_data(show_spinner=False)
def cached_connectivity(landcover_file, threshold_m, metric_crs, core_habitats):
    return calculate_connectivity(landcover_file, float(threshold_m), metric_crs, list(core_habitats))


@st.cache_data(show_spinner=False)
def cached_gbif_df(path):
    return pd.read_csv(path)


# ============================================================================
# ANALYSE GBIF
# ============================================================================

def analyze_gbif(df):
    if df is None or df.empty or "title" not in df.columns:
        return {"usable": False, "message": "Colonne 'title' manquante."}
    data = df.copy()
    data["title"] = data["title"].astype(str).str.strip()
    data = data[data["title"] != ""]
    if data.empty:
        return {"usable": False, "message": "Aucune espece exploitable."}
    total = len(data)
    top = data["title"].value_counts().reset_index()
    top.columns = ["species", "observations"]
    top["part_%"] = top["observations"] / total * 100
    return {
        "usable": True, "total_obs": total,
        "unique_species": data["title"].nunique(), "top_species": top,
        "top1_share": float(top.iloc[0]["part_%"]) if not top.empty else 0.0,
        "top10_share": float(top.head(10)["observations"].sum() / total * 100),
        "raw_df": data,
    }


# ============================================================================
# COMMENTAIRES
# ============================================================================

def _level(val, thresholds, labels):
    for t, l in zip(thresholds, labels):
        if val >= t: return l
    return labels[-1]


def sb_caption(sb_df, global_sb):
    if sb_df is None or sb_df.empty or global_sb is None: return "Score SB non disponible."
    state = _level(global_sb, [75, 60, 45], ["tres bon", "bon mais fragile", "intermediaire", "degrade"])
    top, bot = sb_df.iloc[0], sb_df.iloc[-1]
    return (f"Biodiversite **{state}** — score **{global_sb:.1f}/100**. "
            f"Meilleur habitat : **{top['cover_label']}** ({top['score_sb_100']:.1f}). "
            f"Moins favorable : **{bot['cover_label']}** ({bot['score_sb_100']:.1f}).")


def sp_caption(sp_result):
    if sp_result is None: return "Score SP non disponible."
    sc = sp_result["score_sp"]
    state = _level(sc, [75, 60, 45], ["tres favorable", "favorable", "intermediaire", "degrade"])
    return (f"Paysage **{state}** — score **{sc:.1f}/100**. "
            f"Naturalite : **{sp_result['naturality']:.3f}** · "
            f"Shannon : **{sp_result['shannon_paysage']:.3f}** · "
            f"Pielou : **{sp_result['pielou']:.3f}**.")


def v3_caption_text(global_v3, eff_weights):
    if global_v3 is None: return "Score V3 non disponible."
    state = _level(global_v3, [75, 60, 45], ["tres bon", "bon mais fragile", "intermediaire", "degrade"])
    w = eff_weights
    return (f"Score V3 **{state}** — **{global_v3:.1f}/100**. "
            f"Ponderations : SB {w.get('sb', 0):.0%} · SP {w.get('sp', 0):.0%} · SC {w.get('sc', 0):.0%}.")


def priority_actions(v3_df):
    if v3_df is None or v3_df.empty: return []
    top  = v3_df[v3_df["score_v3_100"] >= 75]["cover_label"].tolist()
    weak = v3_df[v3_df["score_v3_100"] <  45]["cover_label"].tolist()
    out = []
    if top:  out.append("Preserver en priorite : " + ", ".join(top))
    if weak: out.append("Restaurer ou ameliorer : " + ", ".join(weak))
    return out


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown(
        f"<p style='font-size:1rem;font-weight:700;color:{C_NAVY};"
        f"font-family:Georgia,serif;margin-bottom:1rem;'>Analyse ecosysteme</p>",
        unsafe_allow_html=True,
    )
    input_mode  = st.radio("Source des fichiers", ["Upload", "Chemin local"], horizontal=True)
    metric_crs  = st.text_input("CRS metrique", value="EPSG:2154")
    threshold_m = st.number_input("Seuil connectivite (m)", 100, 10000, 1000, 100)
    all_habitats = ["Feuillus", "Prairie", "Eau", "Coniferes", "Pelouse", "Landes", "Vignes", "Vergers"]
    core_habitats = st.multiselect("Habitats coeur (connectivite)", all_habitats, default=DEFAULT_CORE_HABITATS)

    st.divider()
    st.markdown(
        f"<p style='font-size:0.72rem;font-weight:700;color:{C_NAVY};"
        f"text-transform:uppercase;letter-spacing:0.05em;'>Ponderations V3</p>",
        unsafe_allow_html=True,
    )
    w_sb = st.slider("SB — Biodiversite",  0, 100, int(WEIGHTS_V3["sb"] * 100), 5)
    w_sp = st.slider("SP — Paysage",       0, 100, int(WEIGHTS_V3["sp"] * 100), 5)
    w_sc = st.slider("SC — Connectivite",  0, 100, int(WEIGHTS_V3["sc"] * 100), 5)
    w_total = w_sb + w_sp + w_sc
    if w_total > 0:
        custom_weights = {"sb": w_sb / w_total, "sp": w_sp / w_total, "sc": w_sc / w_total}
        st.caption(f"Normalise : SB {custom_weights['sb']:.0%} · SP {custom_weights['sp']:.0%} · SC {custom_weights['sc']:.0%}")
    else:
        st.error("Les ponderations ne peuvent pas toutes etre a 0.")
        custom_weights = WEIGHTS_V3

    st.divider()

    # -- Chemins démo (relatifs à la racine du projet)
    DEMO_GBIF = "datasets/GBIF-lapeyruche.csv"
    DEMO_LC   = "datasets/occ-lapeyruche.geojson"
    demo_available = Path(DEMO_GBIF).exists() and Path(DEMO_LC).exists()

    run_demo   = False
    run_custom = False

    # -- Bouton démo
    if demo_available:
        run_demo = st.button(
            "Lancer la demo (La Peyruche)",
            use_container_width=True,
        )
        st.caption("Données demo : GBIF + occupation du sol du site La Peyruche.")
        st.divider()
    else:
        st.caption(
            "Fichiers demo introuvables dans `datasets/`. "
            "Deposez `GBIF-lapeyruche.csv` et `occ-lapeyruche.geojson` dans le dossier `datasets/` "
            "pour activer la demo."
        )
        st.divider()

    # -- Chargement de fichiers personnalisés
    st.markdown(
        f"<p style='font-size:0.72rem;font-weight:700;color:{C_NAVY};"
        f"text-transform:uppercase;letter-spacing:0.05em;'>Ou chargez vos propres fichiers</p>",
        unsafe_allow_html=True,
    )

    gbif_file_path = landcover_file_path = None
    if input_mode == "Upload":
        gbif_up = st.file_uploader("Observations GBIF (CSV) — optionnel", type=["csv"])
        lc_up   = st.file_uploader("Occupation du sol (GeoJSON) — requis", type=["geojson", "json"])
        if gbif_up: gbif_file_path      = save_file(gbif_up, ".csv")
        if lc_up:   landcover_file_path = save_file(lc_up,   ".geojson")
    else:
        gbif_file_path      = st.text_input("Chemin GBIF CSV (optionnel)", "") or None
        landcover_file_path = st.text_input("Chemin GeoJSON (requis)",     "") or None

    if landcover_file_path:
        run_custom = st.button("Lancer l'analyse", type="primary", use_container_width=True)

    # -- Résolution finale des chemins et du déclencheur
    run = run_demo or run_custom
    if run_demo:
        gbif_file_path      = DEMO_GBIF
        landcover_file_path = DEMO_LC
        is_demo = True
    else:
        is_demo = False


# ============================================================================
# EN-TETE
# ============================================================================

st.markdown(
    f"<h1 style='font-size:1.6rem;margin-bottom:0.1rem;color:{C_TEXT};font-family:Georgia,serif;'>"
    f"Score d'etat de l'ecosysteme</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='font-size:0.8rem;color:#777;margin-top:0;letter-spacing:0.05em;'>"
    f"Score Biodiversite (SB) &nbsp;·&nbsp; Score Paysage (SP) &nbsp;·&nbsp; "
    f"Connectivite (SC) &nbsp;·&nbsp; Synthese V3</p>", unsafe_allow_html=True)


# ============================================================================
# EXECUTION
# ============================================================================

if run:
    errors = []
    if not landcover_file_path: errors.append("Le fichier d'occupation du sol est requis.")
    if not core_habitats:       errors.append("Selectionnez au moins un habitat coeur.")
    if w_total == 0:            errors.append("Les ponderations V3 ne peuvent pas toutes etre a 0.")
    if input_mode == "Chemin local":
        if gbif_file_path and not Path(gbif_file_path).exists():
            errors.append(f"GBIF introuvable : {gbif_file_path}")
        if landcover_file_path and not Path(landcover_file_path).exists():
            errors.append(f"GeoJSON introuvable : {landcover_file_path}")
    if errors:
        for e in errors: st.error(e)
        st.stop()

    with st.spinner("Analyse en cours..."):
        try:
            sp_result = cached_score_sp(landcover_file_path, metric_crs)

            sb_df, gbif_df = pd.DataFrame(), pd.DataFrame()
            if gbif_file_path:
                gbif_df = cached_gbif_df(gbif_file_path)
                sb_df, _ = cached_score_sb(gbif_file_path, landcover_file_path, metric_crs)

            patches_gdf, conn_summary_df, conn_stats = cached_connectivity(
                landcover_file_path, threshold_m, metric_crs, tuple(core_habitats))
            sc_df = compute_score_sc(conn_summary_df)

            v3_df = compute_score_v3_by_habitat(
                sb_df if not sb_df.empty else None, sp_result, sc_df, weights=custom_weights)

            global_sb_val = compute_global_score_sb(sb_df)   if not sb_df.empty else None
            global_sp_val = compute_global_score_sp(sp_result)
            global_sc_val = compute_global_score_sc(sc_df, conn_summary_df)
            global_v3_val = compute_global_score_v3(v3_df)
            eff_weights   = get_effective_weights(v3_df)
            gbif_analysis = analyze_gbif(gbif_df)

            st.session_state.update({
                "sp_result": sp_result, "sb_df": sb_df, "global_sb": global_sb_val,
                "gbif_df": gbif_df, "gbif_analysis": gbif_analysis,
                "patches_gdf": patches_gdf, "conn_summary_df": conn_summary_df,
                "conn_stats": conn_stats, "sc_df": sc_df, "global_sc": global_sc_val,
                "v3_df": v3_df, "global_sp": global_sp_val,
                "global_v3": global_v3_val, "eff_weights": eff_weights,
                "gbif_file_path": gbif_file_path, "lc_path": landcover_file_path,
                "threshold_m": float(threshold_m),
                "is_demo": is_demo,
            })
        except Exception as exc:
            st.error("Erreur pendant le calcul.")
            st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            st.stop()


# ============================================================================
# AFFICHAGE
# ============================================================================

if "v3_df" not in st.session_state:
    st.markdown(f"""
<div style='max-width:620px;margin-top:2rem;'>

<p style='font-size:1rem;color:{C_TEXT};line-height:1.7;'>
Cet outil calcule trois scores complementaires pour evaluer la qualite ecologique d'un territoire :
</p>

<table style='width:100%;font-size:0.88rem;color:{C_TEXT};border-collapse:collapse;margin-bottom:1.2rem;'>
<tr>
  <td style='padding:6px 12px 6px 0;font-weight:700;color:{C_SB};'>SB — Biodiversite</td>
  <td style='padding:6px 0;'>Richesse, densite et rarete des especes observees (GBIF)</td>
</tr>
<tr style='background:#f5f0e8;'>
  <td style='padding:6px 12px 6px 0;font-weight:700;color:{C_SP};'>SP — Paysage</td>
  <td style='padding:6px 0;'>Naturalite paysagere basee sur l'occupation du sol</td>
</tr>
<tr>
  <td style='padding:6px 12px 6px 0;font-weight:700;color:{C_SC};'>SC — Connectivite</td>
  <td style='padding:6px 0;'>Proximite spatiale entre patches d'habitats coeur</td>
</tr>
<tr style='background:#f5f0e8;'>
  <td style='padding:6px 12px 6px 0;font-weight:700;color:{C_V3};'>V3 — Synthese</td>
  <td style='padding:6px 0;'>Score global combine SB + SP + SC, poids ajustables</td>
</tr>
</table>

<p style='font-size:0.88rem;color:#666;'>
Pour commencer, utilisez le panneau lateral :
</p>
<ul style='font-size:0.88rem;color:{C_TEXT};line-height:1.9;margin-top:0;'>
  <li><strong>Lancer la demo</strong> — analyse immediate sur le site de La Peyruche</li>
  <li><strong>Charger vos fichiers</strong> — GeoJSON d'occupation du sol (requis) + CSV GBIF (optionnel)</li>
</ul>

</div>
""", unsafe_allow_html=True)
    st.stop()

sp_result       = st.session_state["sp_result"]
sb_df           = st.session_state["sb_df"]
global_sb       = st.session_state["global_sb"]
gbif_df         = st.session_state["gbif_df"]
gbif_analysis   = st.session_state["gbif_analysis"]
conn_summary_df = st.session_state["conn_summary_df"]
conn_stats      = st.session_state["conn_stats"]
patches_gdf     = st.session_state["patches_gdf"]
sc_df           = st.session_state["sc_df"]
global_sc       = st.session_state["global_sc"]
v3_df           = st.session_state["v3_df"]
global_sp       = st.session_state["global_sp"]
global_v3       = st.session_state["global_v3"]
eff_weights     = st.session_state["eff_weights"]
gbif_file_path  = st.session_state["gbif_file_path"]
lc_path         = st.session_state["lc_path"]

has_sb   = not sb_df.empty and "score_sb_100" in sb_df.columns
has_conn = not conn_summary_df.empty
has_v3   = not v3_df.empty

st.success("Analyse terminee" + (" — mode demo : site La Peyruche" if st.session_state.get("is_demo") else ""))

tabs = st.tabs([
    "Vue d'ensemble", "Score SB - Biodiversite", "Score SP - Paysage",
    "Connectivite SC", "Score V3 - Synthese", "Methode",
])


# ── TAB 1 : Vue d'ensemble ──────────────────────────────────────────────────
with tabs[0]:
    cols = st.columns(4)
    with cols[0]:
        section_header("Score SB - Biodiversite")
        if has_sb:
            st.plotly_chart(gauge(global_sb, "Score SB", C_SB), use_container_width=True)
            st.markdown(status_badge(global_sb) + "&nbsp;&nbsp;<span style='font-size:0.8rem;color:#555;'>Biodiversite GBIF</span>", unsafe_allow_html=True)
        else:
            st.info("GBIF non fourni - SB non calcule.")
    with cols[1]:
        section_header("Score SP - Paysage")
        st.plotly_chart(gauge(global_sp, "Score SP", C_SP), use_container_width=True)
        st.markdown(status_badge(global_sp) + "&nbsp;&nbsp;<span style='font-size:0.8rem;color:#555;'>Naturalite paysagere</span>", unsafe_allow_html=True)
    with cols[2]:
        section_header("Connectivite SC")
        sc_pct = min(global_sc, 100)
        st.plotly_chart(gauge(sc_pct, f"SC ({conn_stats.get('nb_patches', 0)} patches)", C_SC), use_container_width=True)
        st.caption(build_connectivity_comment(conn_summary_df, conn_stats) if has_conn else "")
    with cols[3]:
        section_header("Score V3 - Synthese")
        if has_v3:
            st.plotly_chart(gauge(global_v3, "Score V3", C_V3), use_container_width=True)
            st.markdown(status_badge(global_v3) + "&nbsp;&nbsp;<span style='font-size:0.8rem;color:#555;'>SB + SP + SC</span>", unsafe_allow_html=True)
        else:
            st.info("V3 non disponible.")

    if has_v3:
        st.divider()
        section_header("Priorites de gestion")
        for a in priority_actions(v3_df):
            st.markdown(
                f"<p style='font-size:0.85rem;color:{C_TEXT};margin:0.2rem 0;'>"
                f"<span style='color:{C_NAVY};font-weight:700;margin-right:6px;'>-</span>{a}</p>",
                unsafe_allow_html=True)


# ── TAB 2 : Score SB ────────────────────────────────────────────────────────
with tabs[1]:
    if not has_sb:
        st.info("Score SB non disponible - chargez un fichier GBIF pour activer cet onglet.")
    else:
        st.metric("Score global SB", f"{global_sb:.2f} / 100")
        st.caption(sb_caption(sb_df, global_sb))

        a = gbif_analysis
        if a.get("usable"):
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Observations",     f"{a['total_obs']:,}")
            c2.metric("Especes",          f"{a['unique_species']:,}")
            c3.metric("Espece dominante", f"{a['top1_share']:.1f}%")
            c4.metric("Top 10 especes",   f"{a['top10_share']:.1f}%")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            section_header("Score SB par habitat")
            st.plotly_chart(hbar(sb_df, "score_sb_100", "cover_label",
                                 "Score SB par habitat", color_col="score_sb_100"), use_container_width=True)
        with col2:
            section_header("Richesse specifique")
            st.plotly_chart(hbar(sb_df, "especes_uniques", "cover_label",
                                 "Especes uniques par habitat", color_fixed=C_SB), use_container_width=True)

        fig_r = radar_sb(sb_df)
        if fig_r:
            st.divider()
            section_header("Profil des composantes SB")
            st.plotly_chart(fig_r, use_container_width=True)

        if not gbif_df.empty:
            fig_m = monthly_histogram(gbif_df)
            if fig_m:
                st.divider()
                section_header("Dynamique temporelle")
                st.plotly_chart(fig_m, use_container_width=True)

        if a.get("usable"):
            st.divider()
            col1, col2 = st.columns([3, 2])
            with col1:
                section_header("Top 20 especes")
                top20 = a["top_species"].head(20).sort_values("observations", ascending=True)
                fig_top = go.Figure(go.Bar(
                    x=top20["observations"], y=top20["species"], orientation="h",
                    marker=dict(color=top20["observations"],
                                colorscale=[[0, C_NAVY_LIGHT], [1, C_GREEN_DARK]], showscale=False),
                    marker_line_width=0, hovertemplate="%{y}: %{x} obs.<extra></extra>",
                ))
                fig_top.update_layout(
                    height=460, margin=dict(t=10, b=10, l=10, r=10),
                    xaxis=dict(title="Observations", showgrid=True, gridcolor="#e8e0d4"),
                    yaxis=dict(showgrid=False, tickfont=dict(size=9)), **PLOTLY_BASE,
                )
                st.plotly_chart(fig_top, use_container_width=True)
            with col2:
                if gbif_file_path:
                    section_header("Carte des observations")
                    map_gbif(gbif_file_path)

        st.divider()
        section_header("Tableau detaille SB")
        disp_cols = ["cover_label", "score_sb_100", "surface_km2", "observations",
                     "especes_uniques", "obs_par_km2", "especes_par_km2", "fragmentation_simple", "shannon"]
        disp = sb_df[[c for c in disp_cols if c in sb_df.columns]].rename(columns={
            "cover_label": "Habitat", "score_sb_100": "Score SB", "surface_km2": "km2",
            "observations": "Obs.", "especes_uniques": "Especes", "obs_par_km2": "Obs/km2",
            "especes_par_km2": "Esp/km2", "fragmentation_simple": "Fragmentation", "shannon": "Shannon",
        })
        st.dataframe(disp.round(2), use_container_width=True, hide_index=True)
        st.download_button("Telecharger CSV Score SB",
                           sb_df.to_csv(index=False).encode(), "score_sb.csv", "text/csv")


# ── TAB 3 : Score SP ────────────────────────────────────────────────────────
with tabs[2]:
    sp = sp_result
    cl = sp["classes_df"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score SP",         f"{sp['score_sp']:.2f} / 100")
    c2.metric("Naturalite",       f"{sp['naturality']:.3f}")
    c3.metric("Shannon paysage",  f"{sp['shannon_paysage']:.3f}")
    c4.metric("Pielou",           f"{sp['pielou']:.3f}")
    st.caption(sp_caption(sp))

    if sp["classes_inconnues"]:
        with st.expander(f"Classes sans poids explicite ({len(sp['classes_inconnues'])}) — poids par defaut applique"):
            for c in sp["classes_inconnues"]:
                st.markdown(f"- `{c}`")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        section_header("Repartition surfacique")
        st.plotly_chart(hbar(cl, "part_surface_%", "cover_label",
                             "Part de surface (%)", color_fixed=C_SP), use_container_width=True)
    with col2:
        section_header("Contribution a la naturalite")
        st.plotly_chart(hbar(cl, "contribution_sp", "cover_label",
                             "Contribution SP (proportion x poids)", color_fixed=C_GREEN_DARK),
                        use_container_width=True)

    st.divider()
    section_header("Carte d'occupation des sols")
    map_habitat(lc_path)

    st.divider()
    section_header("Tableau detaille SP")
    disp_sp = cl[[c for c in ["cover_label", "surface_ha", "part_surface_%", "poids_ecologique",
                               "contribution_sp", "nb_polygones", "fragmentation_simple"] if c in cl.columns]].rename(columns={
        "cover_label": "Classe", "surface_ha": "Surface (ha)", "part_surface_%": "%",
        "poids_ecologique": "Poids eco.", "contribution_sp": "Contribution SP",
        "nb_polygones": "Nb polygones", "fragmentation_simple": "Fragmentation",
    })
    st.dataframe(disp_sp.round(3), use_container_width=True, hide_index=True)
    st.download_button("Telecharger CSV Score SP",
                       cl.to_csv(index=False).encode(), "score_sp.csv", "text/csv")


# ── TAB 4 : Connectivite SC ─────────────────────────────────────────────────
with tabs[3]:
    if not has_conn:
        st.info("Aucun resultat de connectivite.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Connectivite moyenne", f"{conn_stats.get('connectivite_moyenne_globale', 0):.2f}")
        c2.metric("Patches analyses",     f"{conn_stats.get('nb_patches', 0)}")
        c3.metric("Distance voisin moy.", f"{conn_stats.get('distance_voisin_moy_globale', 0):.0f} m")
        st.caption(build_connectivity_comment(conn_summary_df, conn_stats))
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            section_header("Connectivite moyenne par habitat")
            st.plotly_chart(hbar(conn_summary_df, "connectivite_moyenne", "cover_label",
                                 "Voisins connectes (moyenne)", color_fixed=C_SC), use_container_width=True)
        with col2:
            section_header("Distance au plus proche voisin")
            st.plotly_chart(hbar(conn_summary_df, "distance_voisin_moy", "cover_label",
                                 "Distance moyenne (m)", color_fixed=C_NAVY), use_container_width=True)

        st.divider()
        section_header("Cartographie")
        map_choice = st.radio("Couche a afficher",
                              ["Patches de connectivite", "Occupation des sols", "Observations GBIF"],
                              horizontal=True, key="map_conn")
        if map_choice == "Patches de connectivite":  map_connectivity(patches_gdf)
        elif map_choice == "Occupation des sols":    map_habitat(lc_path)
        elif gbif_file_path:                         map_gbif(gbif_file_path)
        else: st.info("Aucun fichier GBIF fourni.")

        st.divider()
        section_header("Tableau de synthese SC")
        st.dataframe(conn_summary_df.round(2), use_container_width=True, hide_index=True)
        st.download_button("Telecharger CSV Connectivite",
                           conn_summary_df.to_csv(index=False).encode(), "connectivite.csv", "text/csv")


# ── TAB 5 : Score V3 ────────────────────────────────────────────────────────
with tabs[4]:
    if not has_v3:
        st.info("Score V3 non disponible.")
    else:
        ncols = 4 if has_sb else 3
        cols_v3 = st.columns(ncols)
        idx = 0
        if has_sb:
            cols_v3[idx].metric("Score SB", f"{global_sb:.2f} / 100"); idx += 1
        cols_v3[idx].metric("Score SP",  f"{global_sp:.2f} / 100"); idx += 1
        cols_v3[idx].metric("Score SC",  f"{global_sc:.2f} / 100"); idx += 1
        cols_v3[idx].metric("Score V3",  f"{global_v3:.2f} / 100")
        st.caption(v3_caption_text(global_v3, eff_weights))
        st.divider()

        section_header("Comparaison SB / SP / SC / V3 par habitat")
        st.plotly_chart(v3_comparison(v3_df, has_sb), use_container_width=True)

        st.divider()
        section_header("Tableau synthetique V3")
        v3_cols = ["cover_label", "surface_km2"]
        if has_sb: v3_cols.append("score_sb_100")
        v3_cols += ["score_sp_100", "score_sc_100", "score_v3_100"]
        disp_v3 = v3_df[[c for c in v3_cols if c in v3_df.columns]].rename(columns={
            "cover_label": "Habitat", "surface_km2": "km2",
            "score_sb_100": "SB", "score_sp_100": "SP",
            "score_sc_100": "SC", "score_v3_100": "V3",
        })
        st.dataframe(disp_v3.round(2), use_container_width=True, hide_index=True)
        st.download_button("Telecharger CSV Score V3",
                           v3_df.to_csv(index=False).encode(), "score_v3.csv", "text/csv")


# ── TAB 6 : Methode ─────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown(f"""
<div style='max-width:740px;font-family:Calibri,sans-serif;font-size:0.88rem;color:{C_TEXT};line-height:1.75;'>

<p style='font-size:1rem;font-weight:700;color:{C_SB};font-family:Georgia,serif;'>Score SB — Biodiversite</p>
6 composantes normalisees [0-1] ponderees par habitat — necessite un fichier GBIF.

| Composante | Poids | Description |
|---|---|---|
| Richesse specifique | 25% | Nombre d'especes uniques |
| Densite de richesse | 20% | Especes / km2 |
| Densite d'observations | 15% | Obs. / km2 |
| Fragmentation (inverse) | 15% | Nb polygones / km2 inverse |
| Shannon | 15% | Diversite des especes observees |
| Rarete proxy | 10% | 1 / frequence globale de l'espece |

<p style='font-size:1rem;font-weight:700;color:{C_SP};font-family:Georgia,serif;margin:1.2rem 0 0.5rem;'>Score SP — Paysage</p>
<code>SP = 100 x sum(pi x wi)</code> — proportion surfacique x poids ecologique par classe. Ne necessite pas de GBIF.

<p style='font-size:1rem;font-weight:700;color:{C_SC};font-family:Georgia,serif;margin:1.2rem 0 0.5rem;'>Score SC — Connectivite</p>
<code>SC = 60% rang(connectivite) + 30% inverse(distance) + 10% rang(nb patches)</code>

<p style='font-size:1rem;font-weight:700;color:{C_V3};font-family:Georgia,serif;margin:1.2rem 0 0.5rem;'>Score V3 — Synthese</p>
<code>V3 = w_SB x SB + w_SP x SP + w_SC x SC</code> — ponderations ajustables dans la barre laterale.
Si SB est absent, les poids sont redistribues automatiquement entre SP et SC.

<p style='font-size:1rem;font-weight:700;color:{C_TEXT};font-family:Georgia,serif;margin:1.2rem 0 0.5rem;'>Limites</p>
<ul>
<li>Rarete SB = proxy frequence observee, pas un statut de conservation officiel.</li>
<li>Connectivite = distance euclidienne entre centroides.</li>
<li>Le diagnostic est un outil de pre-analyse, pas un avis reglementaire.</li>
</ul>
</div>
""", unsafe_allow_html=True)
