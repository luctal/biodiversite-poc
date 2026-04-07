import os
import base64
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
# FONCTION : image cliquable
# ---------------------------------------------------------
def clickable_image(image_path, target_url, alt_text="image"):
    """
    Affiche une image cliquable qui renvoie vers une page de l'app.
    Si l'image n'existe pas, affiche un placeholder à la place.
    """
    if not os.path.exists(image_path):
        st.markdown(
            f"""
            <a href="{target_url}" target="_self">
                <div style="
                    width:100%; height:200px; border-radius:8px;
                    background-color:#e8e0d4; display:flex;
                    align-items:center; justify-content:center;
                    cursor:pointer; border: 1px solid #d0c8bc;
                ">
                    <span style="color:#888; font-size:0.85rem;">
                        {alt_text}
                    </span>
                </div>
            </a>
            """,
            unsafe_allow_html=True
        )
        return

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    img_base64 = base64.b64encode(img_bytes).decode()
    ext = image_path.split(".")[-1].lower()
    mime_type = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"

    st.markdown(
        f"""
        <a href="{target_url}" target="_self">
            <img src="data:{mime_type};base64,{img_base64}"
                 alt="{alt_text}"
                 style="width:100%; height:200px; object-fit:cover;
                        border-radius:8px; cursor:pointer;">
        </a>
        """,
        unsafe_allow_html=True
    )


def module_link(target_url, label):
    st.markdown(
        f"""
        <a href="{target_url}" target="_self" style="
            text-decoration: none;
            color: black;
            font-size: 22px;
            font-weight: bold;
        ">
            {label}
        </a>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# TITRE
# ---------------------------------------------------------
st.markdown(
    "<h1 style='font-family:Georgia,serif;'>BioAtlas</h1>",
    unsafe_allow_html=True
)
st.subheader("Choisissez votre module d'analyse")
st.markdown("")

# =========================================================
# LIGNE 1 : CAMERA / SON
# =========================================================
col1, col2 = st.columns(2)

with col1:
    clickable_image("images/chevreuil.jpg", "/app_cam", "Module Camera")
    module_link("/app_cam", "Module Camera")
    st.write("Analyse des observations issues des cameras pieges.")
    st.write("Richesse, activite, comparaisons inter-sites, diagnostic ecologique.")

with col2:
    clickable_image("images/verdier.jpg", "/app_son", "Module Son")
    module_link("/app_son", "Module Son")
    st.write("Analyse des observations acoustiques (BirdNET).")
    st.write("Indices, statistiques, activite, diagnostic ecologique.")

st.markdown("")

# =========================================================
# LIGNE 2 : CHAUVES-SOURIS / SCORE ECOSYSTEME
# =========================================================
col3, col4 = st.columns(2)

with col3:
    clickable_image("images/pipit.JPG", "/app_bat", "Module Chauves-souris")
    module_link("/app_bat", "Module Chauves-souris")
    st.write("Analyse des observations acoustiques ultrasonores.")
    st.write("Activite, diversite, comparaisons inter-sites, diagnostic ecologique.")

with col4:
    clickable_image("images/paysage.jpg", "/Score_Ecosysteme", "Score Ecosysteme")
    module_link("/Score_Ecosysteme", "Score Ecosysteme V3")
    st.write("Score de qualite ecologique du territoire.")
    st.write("Biodiversite (SB), paysage (SP), connectivite (SC) et synthese V3.")
