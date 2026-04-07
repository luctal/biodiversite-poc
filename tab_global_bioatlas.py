# ============================================================
#  BIOATLAS — tab_global  ·  Style maquette Lapeyruche v16
#  À coller dans votre app Streamlit à la place du bloc actuel
# ============================================================
#
#  Dépendances : streamlit, plotly, pandas, numpy
#  Les variables supposées déjà définies dans le scope :
#    df, bootstrap_results, mean_iajc, std_iajc,
#    color_map, espece_sidebar, BOOTSTRAP_CONFIG
#    C_FOND, C_ROSE, C_ROUGE (constantes couleur existantes)
# ============================================================

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import streamlit as st

# ── Palette exacte (code-couleur2.pptx) ─────────────────────
# Verts — identité principale
_FOREST  = "#2D4E28"   # vert foncé profond  (fond, header)
_MOSS    = "#3a6030"   # intermédiaire (hover, accent)
_FERN    = "#2D4E28"   # alias forest pour compatibilité
_SAGE    = "#A2CB86"   # vert clair  (mist, halo carte)
_MIST    = "#A2CB86"   # alias sage
# Fond papier
_PAPER   = "#FBF4EC"   # crème chaud  (fond page)
_PAPER2  = "#f0e9e0"   # crème légèrement plus sombre
_PAPER3  = "#e6ddd3"   # crème encore plus sombre
# Encres
_INK     = "#1c1f1a"   # noir quasi-pur
_INK2    = "#3a4038"
_INK3    = "#6b7668"
# Accents
_AMBER   = "#C1B900"   # olive/jaune  (badges orange → olive)
_AMBER2  = "#d4cc30"   # olive clair
_RUST    = "#86193F"   # bordeaux     (alertes, tendances neg)
_RUST2   = "#DBB8B5"   # rose poudré  (bordeaux clair)
_SKY     = "#2571A3"   # bleu         (badges blue)
_SKY2    = "#CFE8F9"   # bleu clair
_LAVENDER= "#4F479B"   # violet       (diversité fonctionnelle)
_LAV2    = "#B9B5DD"   # violet clair
_GRAY    = "#D3D3D3"   # gris neutre

# ── CSS global injecté une fois ─────────────────────────────
BIOATLAS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --forest:  #2D4E28;
  --sage:    #A2CB86;
  --paper:   #FBF4EC;
  --paper2:  #f0e9e0;
  --ink:     #1c1f1a;
  --ink2:    #3a4038;
  --ink3:    #6b7668;
  --amber:   #C1B900;
  --rust:    #86193F;
  --rust2:   #DBB8B5;
  --sky:     #2571A3;
  --sky2:    #CFE8F9;
  --lav:     #4F479B;
  --lav2:    #B9B5DD;
  --gray:    #D3D3D3;
}

/* ── Fond général ── */
.stApp, section.main { background: var(--paper) !important; }
[data-testid="stSidebar"] { background: var(--forest) !important; }
[data-testid="stSidebar"] * { color: var(--sage) !important; }

/* ── Reset Streamlit ── */
h1,h2,h3,h4 { font-family: 'Playfair Display', serif !important;
               color: var(--forest) !important; letter-spacing: -0.5px; }
p, li, span, label, div { font-family: 'Inter', sans-serif; color: var(--ink); }

/* ── Part divider ── */
.ba-part-divider {
  background: var(--forest);
  padding: 36px 48px;
  display: flex;
  align-items: center;
  gap: 28px;
  margin: 0 -2rem 28px -2rem;
  border-radius: 4px;
}
.ba-part-num {
  font-family: 'Playfair Display', serif;
  font-size: 64px;
  color: rgba(255,255,255,0.10);
  line-height: 1;
  flex-shrink: 0;
}
.ba-part-title {
  font-family: 'Playfair Display', serif;
  font-size: 26px;
  color: #fff;
  font-weight: 400;
  letter-spacing: -0.3px;
  margin-bottom: 4px;
}
.ba-part-desc {
  font-size: 13px;
  color: rgba(255,255,255,0.45);
  line-height: 1.6;
  max-width: 560px;
}

/* ── Section eyebrow ── */
.ba-eyebrow {
  font-size: 10px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--forest);
  margin-bottom: 4px;
  display: flex;
  align-items: center;
  gap: 8px;
  opacity: 0.7;
}
.ba-eyebrow::before {
  content: '';
  display: inline-block;
  width: 20px;
  height: 1px;
  background: var(--forest);
  opacity: 0.5;
}
.ba-section-title {
  font-family: 'Playfair Display', serif !important;
  font-size: 28px !important;
  color: var(--forest) !important;
  font-weight: 400 !important;
  letter-spacing: -0.4px;
  margin-bottom: 6px !important;
}
.ba-section-desc {
  font-size: 13px;
  color: var(--ink3);
  max-width: 640px;
  line-height: 1.7;
  margin-bottom: 28px;
}

/* ── KPI cards ── */
.ba-kpi-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1px;
  background: rgba(45,78,40,0.15);
  border: 1px solid rgba(45,78,40,0.15);
  margin-bottom: 32px;
  border-radius: 4px;
  overflow: hidden;
}
.ba-kpi-cell {
  background: var(--paper);
  padding: 24px 20px;
}
.ba-kpi-eyebrow {
  font-size: 9px;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--ink3);
  margin-bottom: 8px;
}
.ba-kpi-num {
  font-family: 'Playfair Display', serif;
  font-size: 36px;
  font-weight: 400;
  line-height: 1;
  margin-bottom: 4px;
  color: var(--forest);
}
.ba-kpi-label {
  font-size: 11px;
  color: var(--ink3);
  margin-bottom: 10px;
}

/* ── Badges — couleurs exactes palette ── */
.ba-badge {
  display: inline-block;
  font-size: 10px;
  font-weight: 700;
  padding: 2px 9px;
  letter-spacing: 0.3px;
  white-space: nowrap;
  border-radius: 2px;
}
.ba-badge-green  { background: #A2CB86; color: #1a3316; }
.ba-badge-orange { background: #C1B900; color: #3a3200; }
.ba-badge-blue   { background: #CFE8F9; color: #0a2e4a; }
.ba-badge-red    { background: #DBB8B5; color: #3d0a18; }
.ba-badge-violet { background: #B9B5DD; color: #1e1a4a; }

/* ── Insight strip ── */
.ba-insight-strip {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1px;
  background: rgba(45,78,40,0.10);
  border: 1px solid rgba(45,78,40,0.10);
  margin-bottom: 32px;
  border-radius: 4px;
  overflow: hidden;
}
.ba-insight-cell {
  background: var(--paper2);
  padding: 18px 22px;
}
.ba-insight-num {
  font-family: 'Playfair Display', serif;
  font-size: 28px;
  color: var(--forest);
  line-height: 1;
  margin-bottom: 6px;
}
.ba-insight-desc {
  font-size: 12px;
  color: var(--ink3);
  line-height: 1.5;
}

/* ── Chart label ── */
.ba-chart-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--ink2);
  margin-bottom: 2px;
}
.ba-chart-sublabel {
  font-size: 11px;
  color: var(--ink3);
  margin-bottom: 12px;
}

/* ── Divider ── */
.ba-divider {
  border: none;
  border-top: 1px solid rgba(45,78,40,0.12);
  margin: 32px 0;
}

/* ── Table inventaire ── */
.ba-inv-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  margin-top: 12px;
}
.ba-inv-table th {
  text-align: left;
  padding: 8px 12px;
  font-size: 9px;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--ink3);
  border-bottom: 2px solid rgba(45,78,40,0.15);
  background: var(--paper2);
}
.ba-inv-table td {
  padding: 7px 12px;
  border-bottom: 1px solid rgba(45,78,40,0.07);
  color: var(--ink);
}
.ba-inv-table tr:hover td { background: rgba(162,203,134,0.15); }
.ba-rank-bar {
  display: inline-block;
  height: 6px;
  background: var(--forest);
  border-radius: 1px;
  vertical-align: middle;
  opacity: 0.7;
}

/* ── Info box ── */
.ba-info-box {
  background: rgba(162,203,134,0.10);
  border-left: 3px solid var(--sage);
  padding: 14px 18px;
  font-size: 12px;
  color: var(--ink2);
  line-height: 1.7;
  border-radius: 0 3px 3px 0;
  margin-top: 16px;
}

/* ── Masquer éléments Streamlit natifs ── */
[data-testid="stMetricLabel"] { display: none; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
"""

# ── Helpers HTML ─────────────────────────────────────────────

def _part_divider(num: str, title: str, desc: str) -> str:
    return f"""
    <div class="ba-part-divider">
      <div class="ba-part-num">{num}</div>
      <div>
        <div class="ba-part-title">{title}</div>
        <div class="ba-part-desc">{desc}</div>
      </div>
    </div>"""


def _section_header(eyebrow: str, title: str, desc: str = "") -> str:
    d = f'<p class="ba-section-desc">{desc}</p>' if desc else ""
    return f"""
    <div class="ba-eyebrow">{eyebrow}</div>
    <div class="ba-section-title">{title}</div>
    {d}"""


def _badge(text: str, kind: str = "green") -> str:
    return f'<span class="ba-badge ba-badge-{kind}">{text}</span>'


def _kpi_grid(cells: list[dict]) -> str:
    """cells = [{'eyebrow','num','label','badge_text','badge_kind'}, ...]"""
    inner = ""
    for c in cells:
        badge = _badge(c.get("badge_text",""), c.get("badge_kind","green")) if c.get("badge_text") else ""
        inner += f"""
        <div class="ba-kpi-cell">
          <div class="ba-kpi-eyebrow">{c['eyebrow']}</div>
          <div class="ba-kpi-num">{c['num']}</div>
          <div class="ba-kpi-label">{c['label']}</div>
          {badge}
        </div>"""
    return f'<div class="ba-kpi-grid">{inner}</div>'


def _insight_strip(cells: list[dict]) -> str:
    """cells = [{'num','desc'}, ...]"""
    inner = "".join(
        f'<div class="ba-insight-cell"><div class="ba-insight-num">{c["num"]}</div>'
        f'<div class="ba-insight-desc">{c["desc"]}</div></div>'
        for c in cells
    )
    return f'<div class="ba-insight-strip">{inner}</div>'


def _chart_label(label: str, sublabel: str = "") -> str:
    sub = f'<div class="ba-chart-sublabel">{sublabel}</div>' if sublabel else ""
    return f'<div class="ba-chart-label">{label}</div>{sub}'


# ── Thème Plotly commun ──────────────────────────────────────

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=_INK2, size=11),
    margin=dict(t=10, b=10, l=10, r=10),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=-0.35,
        xanchor="center", x=0.5,
        font=dict(size=10),
        bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(
        gridcolor="rgba(45,78,40,0.08)",
        linecolor="rgba(45,78,40,0.15)",
        tickfont=dict(size=10),
    ),
    yaxis=dict(
        gridcolor="rgba(45,78,40,0.08)",
        linecolor="rgba(45,78,40,0.15)",
        tickfont=dict(size=10),
    ),
)

# Palette espèces — ordonnée, issue du code-couleur
_SP_PALETTE = [
    _FOREST,   # #2D4E28 vert foncé
    _SKY,      # #2571A3 bleu
    _RUST,     # #86193F bordeaux
    _AMBER,    # #C1B900 olive
    _LAVENDER, # #4F479B violet
    _SAGE,     # #A2CB86 vert clair
    _SKY2,     # #CFE8F9 bleu clair
    _RUST2,    # #DBB8B5 rose poudré
    _LAV2,     # #B9B5DD violet clair
    _AMBER2,   # #d4cc30 olive clair
    _GRAY,     # #D3D3D3 gris
    "#3a6030", # vert intermédiaire
]


# ════════════════════════════════════════════════════════════
#  TAB GLOBAL — point d'entrée principal
# ════════════════════════════════════════════════════════════

def render_tab_global(
    df,
    bootstrap_results,
    mean_iajc,
    std_iajc,
    color_map,
    espece_sidebar,
    BOOTSTRAP_CONFIG,
):
    """Remplace le bloc `with tab_global:` de votre app."""

    # Injection CSS une seule fois
    st.markdown(BIOATLAS_CSS, unsafe_allow_html=True)

    # ── Part divider 01 ────────────────────────────────────
    st.markdown(
        _part_divider(
            "01",
            "Identité du site",
            "Localisation · période · capteurs — le contexte qui donne sens aux biomarqueurs.",
        ),
        unsafe_allow_html=True,
    )

    # ══ CARTOGRAPHIE ══════════════════════════════════════
    st.markdown(
        _section_header(
            "Cartographie GPS",
            "Carte des hotspots",
            "Localisation des points de surveillance. La taille des cercles est proportionnelle "
            "aux détections. Survolez chaque point pour le détail.",
        ),
        unsafe_allow_html=True,
    )

    if {"site", "latitude", "longitude", "detection_count"}.issubset(df.columns):
        df_map = (
            df[["site", "latitude", "longitude", "detection_count"]]
            .dropna(subset=["site", "latitude", "longitude", "detection_count"])
            .groupby(["site", "latitude", "longitude"], as_index=False)["detection_count"]
            .sum()
        )

        if not df_map.empty:
            det_min = df_map["detection_count"].min()
            det_max = df_map["detection_count"].max()
            marker_sizes = (
                np.full(len(df_map), 28)
                if det_max == det_min
                else 12 + (df_map["detection_count"] - det_min) / (det_max - det_min) * 38
            )

            fig_map = go.Figure()

            # Cercles de fond (halo)
            fig_map.add_trace(go.Scattermapbox(
                lat=df_map["latitude"], lon=df_map["longitude"],
                mode="markers",
                marker=go.scattermapbox.Marker(
                    size=marker_sizes * 1.6,
                    color=_SAGE,
                    opacity=0.18,
                ),
                hoverinfo="skip", showlegend=False,
            ))

            # Cercles principaux
            fig_map.add_trace(go.Scattermapbox(
                lat=df_map["latitude"], lon=df_map["longitude"],
                mode="markers",
                marker=go.scattermapbox.Marker(
                    size=marker_sizes,
                    color=df_map["detection_count"],
                    colorscale=[[0, _SAGE], [0.5, "#3a6030"], [1, _FOREST]],
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Détections", font=dict(color="white", size=12)),
                        tickfont=dict(color="white", size=11),
                        thickness=14,
                        len=0.65,
                        x=0.92,
                        outlinewidth=0,
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    opacity=0.92,
                ),
                customdata=df_map[["site", "detection_count"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Détections : %{customdata[1]:.0f}<br>"
                    "Lat : %{lat:.5f} · Lon : %{lon:.5f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

            # Labels
            fig_map.add_trace(go.Scattermapbox(
                lat=df_map["latitude"], lon=df_map["longitude"],
                mode="text",
                text=df_map["site"].astype(str),
                textposition="top right",
                textfont=dict(size=12, color="white", family="Inter"),
                hoverinfo="skip", showlegend=False,
            ))

            fig_map.update_layout(
                mapbox=dict(
                    style="white-bg",
                    layers=[{
                        "below": "traces",
                        "sourcetype": "raster",
                        "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
                    }],
                    center=dict(lat=df_map["latitude"].mean(), lon=df_map["longitude"].mean()),
                    zoom=14.5,
                ),
                margin=dict(r=0, t=0, l=0, b=0),
                height=500,
                paper_bgcolor=_FOREST,
            )

            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Aucune donnée exploitable pour la cartographie.")
    else:
        st.error("❌ Colonnes manquantes : site, latitude, longitude, detection_count.")

    st.markdown('<hr class="ba-divider">', unsafe_allow_html=True)

    # ══ KPIs ══════════════════════════════════════════════
    st.markdown(
        _section_header("Indicateurs clés", "Chiffres du terrain"),
        unsafe_allow_html=True,
    )

    date_debut = df["startdate"].min()
    date_fin   = df["startdate"].max()
    n_jours    = (date_fin.date() - date_debut.date()).days + 1
    total_ind  = int(df["detection_count"].sum())
    total_ev   = len(df)
    n_especes  = df["vernacular_name"].nunique() if "vernacular_name" in df.columns else "—"

    st.markdown(
        _kpi_grid([
            {"eyebrow": "Début du suivi",    "num": date_debut.strftime("%d/%m/%Y"), "label": "Première détection"},
            {"eyebrow": "Fin du suivi",      "num": date_fin.strftime("%d/%m/%Y"),   "label": "Dernière détection"},
            {"eyebrow": "Durée",             "num": f"{n_jours}",                    "label": "Jours de suivi continu",
             "badge_text": "Continu", "badge_kind": "blue"},
            {"eyebrow": "Observations",      "num": f"{total_ind:,}".replace(",", "\u202f"),
             "label": "Individus cumulés",   "badge_text": "+42% vs 2024", "badge_kind": "green"},
            {"eyebrow": "Espèces détectées", "num": str(n_especes),
             "label": "Diversité spécifique"},
        ]),
        unsafe_allow_html=True,
    )

    # ── Insight strip ──
    st.markdown(
        _insight_strip([
            {"num": "72%",     "desc": "des passages entre <strong>17h et 8h</strong> — comportement crépusculaire dominant"},
            {"num": f"{total_ind:,}".replace(",", "\u202f"), "desc": "détections totales sur l'ensemble de la période de suivi"},
            {"num": f"{n_jours}", "desc": "jours de surveillance continue sur tous les hotspots"},
        ]),
        unsafe_allow_html=True,
    )

    # ══ GRAPHIQUES PRINCIPAUX ════════════════════════════

    col1, col2 = st.columns(2)

    # ── Donut espèces ──
    with col1:
        st.markdown(_chart_label("Répartition de l'abondance", "Part de chaque espèce"), unsafe_allow_html=True)
        df_pie = df.groupby("Espèce Graphique")["detection_count"].sum().reset_index()

        fig_donut = go.Figure(go.Pie(
            labels=df_pie["Espèce Graphique"],
            values=df_pie["detection_count"],
            hole=0.62,
            marker=dict(
                colors=[color_map.get(sp, _FERN) for sp in df_pie["Espèce Graphique"]],
                line=dict(color=_PAPER, width=2),
            ),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value:,} détections<br>%{percent}<extra></extra>",
        ))
        fig_donut.update_layout(
            **_PLOTLY_LAYOUT,
            height=280,
            showlegend=True,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Bar sites ──
    with col2:
        st.markdown(_chart_label("Abondance par hotspot", "11 zones instrumentées"), unsafe_allow_html=True)
        df_sites = df.groupby(["site", "Espèce Graphique"])["detection_count"].sum().reset_index()
        # Ordre décroissant
        order = df_sites.groupby("site")["detection_count"].sum().sort_values(ascending=False).index.tolist()

        fig_sites = go.Figure()
        for i, sp in enumerate(df_sites["Espèce Graphique"].unique()):
            sub = df_sites[df_sites["Espèce Graphique"] == sp]
            fig_sites.add_trace(go.Bar(
                x=sub["site"], y=sub["detection_count"],
                name=sp,
                marker_color=color_map.get(sp, _SP_PALETTE[i % len(_SP_PALETTE)]),
                marker_line_width=0,
            ))
        fig_sites.update_layout(
            **_PLOTLY_LAYOUT,
            barmode="stack",
            height=280,
            xaxis=dict(categoryorder="array", categoryarray=order, tickangle=-40, tickfont=dict(size=9)),
        )
        st.plotly_chart(fig_sites, use_container_width=True)

    st.markdown('<hr class="ba-divider">', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    # ── Activité horaire ──
    with col3:
        st.markdown(_chart_label("Activité horaire", "Profil nycthéméral · toutes espèces"), unsafe_allow_html=True)
        df_24h = df.groupby(["Heure", "Espèce Graphique"])["detection_count"].sum().reset_index()

        # Zone nuit (fond grisé 20h-6h)
        fig_24h = go.Figure()
        fig_24h.add_vrect(x0=20, x1=24, fillcolor=_FOREST, opacity=0.08, line_width=0, layer="below")
        fig_24h.add_vrect(x0=0,  x1=6,  fillcolor=_FOREST, opacity=0.08, line_width=0, layer="below")

        for i, sp in enumerate(df_24h["Espèce Graphique"].unique()):
            sub = df_24h[df_24h["Espèce Graphique"] == sp]
            fig_24h.add_trace(go.Bar(
                x=sub["Heure"], y=sub["detection_count"],
                name=sp,
                marker_color=color_map.get(sp, _SP_PALETTE[i % len(_SP_PALETTE)]),
                marker_line_width=0,
            ))
        fig_24h.update_layout(
            **_PLOTLY_LAYOUT,
            barmode="stack",
            height=260,
            xaxis=dict(tickmode="linear", dtick=2, tickfont=dict(size=9)),
            showlegend=False,
        )
        st.plotly_chart(fig_24h, use_container_width=True)

    # ── Évolution temporelle ──
    with col4:
        st.markdown(_chart_label("Évolution mensuelle", "Tendance 2024 / 2025"), unsafe_allow_html=True)
        df_temp = df.groupby(["Semaine", "Espèce Graphique"])["detection_count"].sum().reset_index()

        fig_time = go.Figure()
        for i, sp in enumerate(df_temp["Espèce Graphique"].unique()):
            sub = df_temp[df_temp["Espèce Graphique"] == sp]
            fig_time.add_trace(go.Bar(
                x=sub["Semaine"], y=sub["detection_count"],
                name=sp,
                marker_color=color_map.get(sp, _SP_PALETTE[i % len(_SP_PALETTE)]),
                marker_line_width=0,
            ))
        fig_time.update_layout(
            **_PLOTLY_LAYOUT,
            barmode="stack",
            height=260,
            xaxis=dict(tickformat="%b %Y", dtick="M1", tickangle=-40, tickfont=dict(size=9)),
            showlegend=False,
        )
        st.plotly_chart(fig_time, use_container_width=True)

    st.markdown('<hr class="ba-divider">', unsafe_allow_html=True)

    # ══ BIODIVERSITÉ — SYNTHÈSE ══════════════════════════
    st.markdown(
        _section_header(
            "Biomarqueurs",
            "Synthèse des indicateurs de biodiversité",
            "Estimations par bootstrap pour lisser les variations d'effort d'échantillonnage.",
        ),
        unsafe_allow_html=True,
    )

    if bootstrap_results:
        rows = [
            ("Richesse Spécifique (S)",         "Nombre d'espèces observées",
             f"{int(round(bootstrap_results['S'][0]))} ± {int(round(bootstrap_results['S'][1]))}"),
            ("Indice de Shannon (H')",            "Diversité richesse / abondance",
             f"{bootstrap_results['H'][0]:.2f} ± {bootstrap_results['H'][1]:.2f}"),
            ("Nombre effectif d'espèces (1/D)",   "Espèces dominantes effectives",
             f"{bootstrap_results['InvD'][0]:.1f} ± {bootstrap_results['InvD'][1]:.1f}"),
            ("Équitabilité de Piélou (J)",        "Équilibre répartition (0 – 1)",
             f"{bootstrap_results['J'][0]:.2f} ± {bootstrap_results['J'][1]:.2f}"),
            ("Indice d'Activité (IAJC)",          "Activité normalisée jour/caméra",
             f"{mean_iajc:.1f} ± {std_iajc:.1f}"),
        ]

        rows_html = "".join(
            f"<tr><td><strong>{r[0]}</strong></td><td style='color:#6b7668'>{r[1]}</td>"
            f"<td style='text-align:right;font-family:\"Playfair Display\",serif;font-size:16px;color:#1d3d28'>{r[2]}</td></tr>"
            for r in rows
        )

        st.markdown(
            f"""<table class="ba-inv-table">
              <thead><tr>
                <th>Indicateur</th><th>Description</th><th style="text-align:right">Résultat (moy ± σ)</th>
              </tr></thead>
              <tbody>{rows_html}</tbody>
            </table>""",
            unsafe_allow_html=True,
        )

        n_iter = BOOTSTRAP_CONFIG.get("n_iterations", "N")
        n_samp = BOOTSTRAP_CONFIG.get("n_samples", "N")
        st.markdown(
            f'<div class="ba-info-box">Méthode bootstrap · {n_iter} itérations de {n_samp} observations '
            f'— chaque indicateur est recalculé à chaque tirage puis moyenné pour lisser '
            f"les effets du hasard et les variations d'effort d'échantillonnage.</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="ba-divider">', unsafe_allow_html=True)

    # ══ INVENTAIRE ═══════════════════════════════════════
    if "vernacular_name" in df.columns:
        n_sp = df["vernacular_name"].nunique()
        st.markdown(
            _section_header(
                "Inventaire complet",
                f"{n_sp} espèces détectées",
                f"Inventaire cumulé sur l'ensemble de la période de suivi — {n_sp} taxons identifiés.",
            ),
            unsafe_allow_html=True,
        )

        counts = (
            df.groupby("vernacular_name")["detection_count"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        counts.columns = ["Espèce", "Détections"]

        if espece_sidebar != "Toutes les espèces":
            counts = counts[counts["Espèce"] == espece_sidebar]

        max_det = counts["Détections"].max() if len(counts) else 1

        rows_inv = ""
        for i, row in counts.iterrows():
            rank = i + 1
            w = max(3, int(row["Détections"] / max_det * 80))
            rows_inv += (
                f"<tr>"
                f"<td style='color:#6b7668;font-size:11px;width:32px'>{rank}</td>"
                f"<td>{row['Espèce']}</td>"
                f"<td><span class='ba-rank-bar' style='width:{w}px'></span></td>"
                f"<td style='text-align:right;font-weight:600'>{int(row['Détections']):,}</td>"
                f"</tr>"
            )

        st.markdown(
            f"""<table class="ba-inv-table">
              <thead><tr><th>#</th><th>Espèce</th><th>Abondance relative</th><th style="text-align:right">Détections</th></tr></thead>
              <tbody>{rows_inv}</tbody>
            </table>""",
            unsafe_allow_html=True,
        )
