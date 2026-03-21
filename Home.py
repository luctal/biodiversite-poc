import streamlit as st

# ---------------------------------------------------------
# CONFIG PAGE
# ---------------------------------------------------------
st.set_page_config(page_title="BioAtlas", layout="wide")

# ---------------------------------------------------------
# CHARTE GRAPHIQUE
# ---------------------------------------------------------
C_FOND = "#FBF4EC"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {C_FOND} !important;
    }}
    .card {{
        padding: 28px;
        border-radius: 18px;
        background: white;
        border: 1px solid #e9e2d8;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        min-height: 220px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# TITRE
# ---------------------------------------------------------
st.title("🍃 BioAtlas")
st.subheader("Choisissez votre module d’analyse")

st.markdown("")

# ---------------------------------------------------------
# CARTES DE NAVIGATION
# ---------------------------------------------------------
col1, col2 = st.columns(2)

# -------- CAMÉRA --------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📷 Module Caméra")
    st.write("Analyse des observations issues des caméras pièges.")
    st.write("Richesse, activité, comparaisons inter-sites, diagnostic écologique.")
    
    st.page_link(
        "pages/app_cam.py",
        label="Ouvrir le module Caméra",
        icon="📷"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

# -------- SON --------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎧 Module Son")
    st.write("Analyse des observations acoustiques (BirdNET).")
    st.write("Indices, statistiques, activité, diagnostic écologique.")
    
    st.page_link(
        "pages/app_son.py",
        label="Ouvrir le module Son",
        icon="🎧"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)