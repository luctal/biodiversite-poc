import streamlit as st
import pandas as pd
import plotly.express as px

# 1. CONFIGURATION
st.set_page_config(page_title="Focus Global", layout="wide")

# 2. CHARTE GRAPHIQUE
C_FOND = "#FBF4EC"
C_ROUGE = "#86193F"
C_JAUNE = "#C1B900"
C_VERT_CLAIR = "#A2CB86"

st.markdown(f"<style>.stApp {{ background-color: {C_FOND} !important; }}</style>", unsafe_allow_html=True)

# 3. DONNÉES (BOSTON Sauvages est supprimé ici)
df_bench = pd.DataFrame([
    {"Site": "SITE", "S": 18.5, "Err": 1.2},
    {"Site": "Etrechy", "S": 10.5, "Err": 0.5},
    {"Site": "Lavallière", "S": 16.8, "Err": 0.9},
    {"Site": "La Peyruche", "S": 19.2, "Err": 1.1}
]).sort_values('S', ascending=True)

# 4. LOGIQUE DE COULEUR (Uniquement Global et les autres)
colors_list = []
for site in df_bench['Site']:
    if site == "SITE":
        colors_list.append(C_ROUGE)
    else:
        colors_list.append(C_VERT_CLAIR)

# 5. CRÉATION DU GRAPHIQUE
fig_bar_h = px.bar(
    df_bench,
    x="S",
    y="Site",
    error_x="Err",
    text="S",
    orientation='h',
    template="none"
)

fig_bar_h.update_traces(
    marker_color=colors_list,
    texttemplate='<b>%{text:.1f}</b>',
    textposition='inside',
    insidetextanchor='middle',
    textfont=dict(color='white', size=16),
    error_x=dict(thickness=3, width=10, color='black')
)

# 6. PERSONNALISATION DES LABELS
fig_bar_h.update_layout(
    paper_bgcolor=C_FOND,
    plot_bgcolor=C_FOND,
    xaxis_title="Richesse Spécifique (S) →",
    yaxis_title=None,
    height=450,
    margin=dict(l=220, r=50, t=50, b=50),
    xaxis=dict(gridcolor='rgba(0,0,0,0.1)', range=[0, df_bench['S'].max() * 1.1]),
    yaxis=dict(
        tickfont=dict(size=16, family="Arial black"),
        ticksuffix="      ",
        automargin=True,
        tickvals=df_bench['Site'],
        ticktext=[
            f'<span style="color:{color}">{site}</span>'
            for site, color in zip(df_bench['Site'], colors_list)
        ]
    )
)

st.plotly_chart(fig_bar_h, use_container_width=True)