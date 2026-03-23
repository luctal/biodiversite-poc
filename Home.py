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
    Si l'image n'existe pas, affiche un message à la place.
    """
    if not os.path.exists(image_path):
        st.info(f"Image introuvable : {image_path}")
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
                 style="width:100%; border-radius:8px; cursor:pointer;">
        </a>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# TITRE
# ---------------------------------------------------------
st.title("🍃 BioAtlas")
st.subheader("Choisissez votre module d’analyse")

st.markdown("")

# =========================================================
# LIGNE 1 : CAMÉRA / SON
# =========================================================
col1, col2 = st.columns(2)

# -------- CAMÉRA --------
with col1:
    clickable_image("images/chevreuil.jpg", "/app_cam", "Module Caméra")

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
    clickable_image("images/verdier.jpg", "/app_son", "Module Son")

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

# =========================================================
# LIGNE 2 : CHAUVES-SOURIS / GBIF
# =========================================================
col3, col4 = st.columns(2)

# -------- CHAUVES-SOURIS --------
with col3:
    bat_img_path = "images/pipit.JPG"

    if os.path.exists(bat_img_path):
        clickable_image(bat_img_path, "/app_bat", "Module Chauves-souris")
    else:
        st.info("Image chauves-souris à ajouter dans le dossier images.")

    st.markdown(
        """
        <a href="/app_bat" target="_self" style="
            text-decoration: none;
            color: black;
            font-size: 28px;
            font-weight: bold;
        ">
            🦇 Module Chauves-souris
        </a>
        """,
        unsafe_allow_html=True
    )

    st.write("Analyse des observations acoustiques ultrasonores.")
    st.write("Activité, diversité, comparaisons inter-sites, diagnostic écologique.")

# # -------- GBIF --------
# with col4:
#     gbif_img_path = "images/GBIF.jpg"

#     if os.path.exists(gbif_img_path):
#         clickable_image(gbif_img_path, "/app_GBIF", "Module GBIF")
#     else:
#         st.info("Image GBIF à ajouter dans le dossier images.")

#     st.markdown(
#         """
#         <a href="/app_GBIF" target="_self" style="
#             text-decoration: none;
#             color: black;
#             font-size: 28px;
#             font-weight: bold;
#         ">
#             🌍 Module GBIF
#         </a>
#         """,
#         unsafe_allow_html=True
#     )

#     st.write("Analyse des données issues de GBIF.")
#     st.write("Occurrences, richesse spécifique, cartographie et exploration des données.")