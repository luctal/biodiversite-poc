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
    st.image("images/chevreuil.jpg", use_container_width=True)

    st.markdown(
    """
    <a href="/app_cam" target="_self" style="
        text-decoration: none;
        color: black;
        font-size: 28px;
        font-weight: bold;
    ">
        📷 Module Caméra
    </a>
    """,
    unsafe_allow_html=True
)

    st.write("Analyse des observations issues des caméras pièges.")
    st.write("Richesse, activité, comparaisons inter-sites, diagnostic écologique.")

# -------- SON --------
with col2:
    st.image("images/verdier.jpg", use_container_width=True)

    st.markdown(
    """
    <a href="/app_son" target="_self" style="
        text-decoration: none;
        color: black;
        font-size: 28px;
        font-weight: bold;
    ">
        🎧 Module Son
    </a>
    """,
    unsafe_allow_html=True
)

    st.write("Analyse des observations acoustiques (BirdNET).")
    st.write("Indices, statistiques, activité, diagnostic écologique.")