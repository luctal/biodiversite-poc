import streamlit as st
import pandas as pd
import comp
import plotly.express as px
import numpy as np
import scikit_posthocs as sp
import scipy.stats as scipy_stats
import io

from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison

# =========================================================
# PARAMÈTRES GLOBAUX - BOOTSTRAP
# =========================================================

BOOTSTRAP_CONFIG = {
    "n_samples": 5000,      # taille de l’échantillon tiré à chaque itération
    "n_iterations": 400,    # nombre de répétitions bootstrap
    "iajc_iterations": 400  # spécifique IAJC (peut être différent si besoin)
}

# --- LÉGENDE DES SITES DE RÉFÉRENCE ---
LEGENDE_SITES = (
    "Références écologiques : "
    "ET = Etrechy (zone périurbaine) · "
    "LV = Lavallière (parc d’hôtel) · "
    "LP = La Peyruche (vignoble bio)"
)

# =========================================================
# RÉFÉRENCES TERRAIN - INDICE E1C SON
# Calibration V1 basée sur 4 sites de référence
# - Etréchy    = bas
# - Lavallière = intermédiaire
# - La Peyruche = excellent
# - Purcari    = bon
# =========================================================
E1C_REFERENCE_SOUND = {
    "Shannon": {
        "min": 3.29,   # Lavallière
        "mid": 3.32,   # La Peyruche
        "good": 3.58,  # Purcari
        "max": 3.65    # pas encore observé
    },
    "Pielou": {
        "min": 0.71,
        "mid": 0.73,
        "good": 0.75,
        "max": 0.77
    },
    "Simpson": {
        "min": 12.7,
        "mid": 16.2,
        "good": 27.9,
        "max": 16.2
    }
}

E1C_WEIGHTS_SOUND = {
    "Shannon": 0.18,
    "Pielou": 0.05,
    "Stabilite": 0.45,
    "Simpson": 0.32
}

E1C_THRESHOLDS_SOUND = {
    "low": 40,
    "medium": 60,
    "high": 80
}

DIAG_THRESHOLDS_SOUND = {
    "dominance_good": 0.50,
    "dominance_medium": 0.70,
    "cv_stable": 0.30,
    "cv_medium": 0.50,
    "nocturnite_low": 50,
    "nocturnite_medium": 70
}


# =========================================================
# CHARGEMENT DES DONNEES DE REFERENCE
# =========================================================

def load_references_indices(path):
    import json

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Ignore _comment
        data = {k: v for k, v in data.items() if not str(k).startswith("_")}

        return data

    except Exception as e:
        st.error(f"Erreur chargement références : {e}")
        st.stop()


# 👇 IMPORTANT : EN DEHORS DE LA FONCTION
REFERENCES_INDICES = load_references_indices(
    "datasets/references_indices_sound.json"
)
         


# 1. CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Bio-Data POC", layout="wide")

# CHARTE GRAPHIQUE
C_FOND = "#FBF4EC"
C_VERT_SOMBRE = "#2D4E28"
C_VERT_CLAIR = "#A2CB86"
C_JAUNE = "#C1B900"
C_GRIS = "#D1D3D4"
C_BLEU = "#2571A3"
C_BLEU_CLAIR = "#CFE8F9"
C_ROSE = "#DBB8B5"
C_ROUGE = "#86193F"
C_VIOLET = "#4F479B"
C_MAUVE = "#B9B5DD"

PALETTE_ESPECES = [C_VERT_SOMBRE, C_VERT_CLAIR, C_JAUNE, C_BLEU, C_VIOLET, C_ROUGE, C_ROSE, C_MAUVE, C_BLEU_CLAIR]

st.markdown(f"<style>.stApp {{ background-color: {C_FOND} !important; }}</style>", unsafe_allow_html=True)

# =========================================================
# FONCTIONS CACHÉES (PERFORMANCE)
# =========================================================

@st.cache_data
def compute_bootstrap_stats(df_in, n_samples, n_iterations):
    return bootstrap_stats(
        df_in,
        n_samples=n_samples,
        n_iterations=n_iterations
    )

@st.cache_data
def compute_bootstrap_iajc(df_in, n_iterations):
    return bootstrap_iajc(
        df_in,
        n_iterations=n_iterations
    )


# 2. FONCTIONS DE CHARGEMENT
@st.cache_data
def load_data(uploaded_file):
    """
    Charge le CSV utilisateur, harmonise les noms de colonnes,
    sécurise les types, puis crée les colonnes temporelles
    standardisées qui seront réutilisées dans tout le script.

    Objectifs :
    - accepter plusieurs variantes de noms de colonnes
    - éviter les plantages si certaines colonnes sont absentes
    - définir UNE logique temporelle commune pour tout le dashboard
    """
    df = pd.read_csv(uploaded_file, sep=None, engine='python')



    # ---------------------------------------------------------
    # 1. Harmonisation des noms de colonnes
    # ---------------------------------------------------------
    # On regroupe ici les variantes possibles d'un même champ.
    # Exemple :
    # - "Common Name"
    # - "Common_name"
    # - "Nom vernaculaire"
    # deviennent tous : "vernacular_name"
    # ---------------------------------------------------------
    mapping = {
        'Nom vernaculaire': 'vernacular_name',
        'Common Name': 'vernacular_name',
        'Common_name': 'vernacular_name',
        'Common name': 'vernacular_name',
        'vernacular_name': 'vernacular_name',

        'Nom scientifique': 'scientific_name',
        'Scientific name': 'scientific_name',
        'Scientifique_name': 'scientific_name',
        'scientific_name': 'scientific_name',

        'Hotspot': 'site',
        'Site': 'site',
        'site': 'site',

        'Indice de confiance BirdNet': 'Birdnet_confidence_index'
        
    }

    df = df.rename(columns=mapping)

    # ---------------------------------------------------------
    # 2. Vérifications minimales
    # ---------------------------------------------------------
    # Sans colonne espèce, on ne peut pas calculer :
    # richesse, Shannon, Piélou, Simpson, etc.
    # Sans startdate, on ne peut pas faire
    # d'analyses temporelles ni d'activité journalière.
    # ---------------------------------------------------------
    if 'vernacular_name' not in df.columns:
        st.error(f"⚠️ Impossible de trouver la colonne des espèces. Colonnes présentes : {list(df.columns)}")
        st.info("💡 Assurez-vous que votre fichier contient une colonne nommée 'Common Name', 'Nom vernaculaire' ou 'vernacular_name'.")
        st.stop()

    if 'startdate' not in df.columns:
        st.error("⚠️ Impossible de trouver la colonne 'startdate'.")
        st.stop()

    # ---------------------------------------------------------
    # 3. Fallback si aucun site n'est fourni
    # ---------------------------------------------------------
    # Certains exports ne contiennent pas de colonne 'site'.
    # Pour éviter que le script plante partout,
    # on crée un site unique par défaut.
    # ---------------------------------------------------------
    if 'site' not in df.columns:
        df['site'] = "Site unique"

    # ---------------------------------------------------------
    # 4. Gestion de detection_count
    # ---------------------------------------------------------
    # Si le fichier ne contient pas de comptage explicite,
    # on considère qu'une ligne = 1 détection.
    # ---------------------------------------------------------
    if 'detection_count' not in df.columns:
        df['detection_count'] = 1

    # Conversion robuste en numérique
    # Les valeurs invalides deviennent NaN puis sont remplacées par 1
    df['detection_count'] = pd.to_numeric(df['detection_count'], errors='coerce').fillna(1)

    # ---------------------------------------------------------
    # 5. Sécurisation des coordonnées géographiques
    # ---------------------------------------------------------
    # Les cartes exigent des latitudes / longitudes numériques.
    # ---------------------------------------------------------
    if 'latitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')

    if 'longitude' in df.columns:
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    # ---------------------------------------------------------
    # 6. Conversion de la date
    # ---------------------------------------------------------
    # Toute date illisible est transformée en NaT puis supprimée.
    # ---------------------------------------------------------
    df['startdate'] = pd.to_datetime(df['startdate'], errors='coerce')
    df = df.dropna(subset=['startdate'])

    # ---------------------------------------------------------
    # 7. Colonnes temporelles standardisées
    # ---------------------------------------------------------
    # On crée ici une seule logique temporelle réutilisable
    # partout dans le script :
    #
    # - Heure      : activité circadienne
    # - week_start : affichage temporel (graphiques)
    # - week_id    : identifiant stable pour stats hebdo
    # - year       : info complémentaire si besoin
    # - iso_week   : numéro de semaine ISO
    # ---------------------------------------------------------
    df['Heure'] = df['startdate'].dt.hour
    df['week_start'] = df['startdate'].dt.to_period('W').apply(lambda r: r.start_time)
    df['week_id'] = df['startdate'].dt.strftime('%G-%V')
    df['year'] = df['startdate'].dt.year
    df['iso_week'] = df['startdate'].dt.isocalendar().week.astype(int)

    return df


@st.cache_data
def load_comparison_data():
    path = "datasets/20260303-indices-sites.csv"
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def build_sites_map_figure_simple(df_input, zoom=12):
    """
    Construit une carte satellite simple de repérage des hotspots.

    Cette carte est différente de la carte analytique principale :
    - taille des points fixe
    - pas de variation selon le nombre de détections
    - nom du hotspot affiché en permanence
    - aucune barre latérale de niveau de détection

    Parameters
    ----------
    df_input : pd.DataFrame
        Jeu de données filtré ou non
    zoom : int or float
        Niveau de zoom initial

    Returns
    -------
    plotly.graph_objects.Figure or None
        Figure prête à afficher, ou None si les colonnes nécessaires
        sont absentes
    """
    import plotly.graph_objects as go

    # ---------------------------------------------------------
    # 1. Vérification des colonnes minimales
    # ---------------------------------------------------------
    required_cols = {'site', 'latitude', 'longitude'}
    if not required_cols.issubset(df_input.columns):
        return None

    # ---------------------------------------------------------
    # 2. Copie + sécurisation des coordonnées
    # ---------------------------------------------------------
    df_map = df_input.copy()
    df_map['latitude'] = pd.to_numeric(df_map['latitude'], errors='coerce')
    df_map['longitude'] = pd.to_numeric(df_map['longitude'], errors='coerce')

    # ---------------------------------------------------------
    # 3. Un seul point par hotspot
    # ---------------------------------------------------------
    df_map = (
        df_map[['site', 'latitude', 'longitude']]
        .dropna(subset=['site', 'latitude', 'longitude'])
        .drop_duplicates(subset=['site', 'latitude', 'longitude'])
    )

    if df_map.empty:
        return None

    # ---------------------------------------------------------
    # 4. Création de la figure
    # ---------------------------------------------------------
    fig_map = go.Figure()

    # Points fixes
    fig_map.add_trace(go.Scattermapbox(
        lat=df_map['latitude'],
        lon=df_map['longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=15,          # taille fixe
            color=C_ROUGE,    # couleur fixe
            opacity=0.90
        ),
        customdata=df_map[['site']],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Lat : %{lat:.5f}<br>"
            "Lon : %{lon:.5f}"
            "<extra></extra>"
        ),
        showlegend=False
    ))

    # Noms affichés en permanence
    fig_map.add_trace(go.Scattermapbox(
        lat=df_map['latitude'],
        lon=df_map['longitude'],
        mode='text',
        text=df_map['site'].astype(str),
        textposition='top right',
        textfont=dict(
            size=13,
            color='white',
            family='Arial Black'
        ),
        hoverinfo='skip',
        showlegend=False
    ))

    # Mise en forme
    fig_map.update_layout(
        mapbox=dict(
            style="white-bg",
            layers=[{
                "below": "traces",
                "sourcetype": "raster",
                "source": [
                    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ]
            }],
            center=dict(
                lat=float(df_map['latitude'].mean()),
                lon=float(df_map['longitude'].mean())
            ),
            zoom=zoom
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        height=450,
        paper_bgcolor=C_FOND,
        plot_bgcolor=C_FOND
    )

    return fig_map

def show_sites_map_popover(df_input, label="📍 Voir la carte des sites", zoom=12):
    """
    Affiche une carte des sites dans une popover à la demande.
    """
    with st.popover(label, use_container_width=False):
        st.markdown("#### Localisation des sites / hotspots")

        fig_map = build_sites_map_figure_simple(df_input, zoom=zoom)

        if fig_map is None:
            st.info("Carte indisponible : colonnes 'site', 'latitude' et 'longitude' nécessaires.")
        else:
            st.plotly_chart(fig_map, use_container_width=True)

# =========================================================
# POPUP DE CHARGEMENT DES DONNÉES
# =========================================================

DEMO_FILE_SON = "datasets/son_demo.csv"

if "raw_df_loaded_son" not in st.session_state:
    st.session_state.raw_df_loaded_son = None

if "df_bench_loaded_son" not in st.session_state:
    st.session_state.df_bench_loaded_son = None

if "dataset_name_loaded_son" not in st.session_state:
    st.session_state.dataset_name_loaded_son = None


@st.dialog("Choisir un dataset CSV")
def open_dataset_dialog():
    st.markdown("### Source de données")
    st.write("Choisissez soit la démo intégrée, soit un fichier CSV de votre ordinateur.")

    col1, col2 = st.columns(2)

    # ---------------------------------------------------------
    # OPTION 1 : LANCER LA DÉMO
    # ---------------------------------------------------------
    with col1:
        st.markdown("#### Démo")
        st.caption("Charge automatiquement le fichier de démonstration son.")

        if st.button("Lancer la démo", use_container_width=True, key="launch_demo_son"):
            try:
                raw_df = load_data(DEMO_FILE_SON)
                df_bench = load_comparison_data()

                st.session_state.raw_df_loaded_son = raw_df
                st.session_state.df_bench_loaded_son = df_bench
                st.session_state.dataset_name_loaded_son = "son_demo.csv"

                st.rerun()

            except Exception as e:
                st.error(f"Erreur lors du chargement de la démo : {e}")

    # ---------------------------------------------------------
    # OPTION 2 : TÉLÉCHARGER / IMPORTER UN CSV
    # ---------------------------------------------------------
    with col2:
        st.markdown("#### Télécharger")
        st.caption("Choisissez un fichier CSV sur votre ordinateur.")

        uploaded_file_popup = st.file_uploader(
            "Sélectionner un CSV",
            type=["csv"],
            key="sidebar_popup_uploader_son"
        )

        if uploaded_file_popup is not None:
            try:
                raw_df = load_data(uploaded_file_popup)
                df_bench = load_comparison_data()

                st.session_state.raw_df_loaded_son = raw_df
                st.session_state.df_bench_loaded_son = df_bench
                st.session_state.dataset_name_loaded_son = uploaded_file_popup.name

                st.rerun()

            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")

# 3. SIDEBAR - CHARGEMENT
st.sidebar.title("📁 Données")

if st.sidebar.button("Charger un dataset CSV", use_container_width=True):
    open_dataset_dialog()

if st.session_state.dataset_name_loaded_son is not None:
    st.sidebar.success(f"Dataset actif : {st.session_state.dataset_name_loaded_son}")

if st.session_state.raw_df_loaded_son is not None:
    raw_df = st.session_state.raw_df_loaded_son
    df_bench = st.session_state.df_bench_loaded_son
else:
    st.info("👋 Veuillez charger un fichier CSV ou lancer la démo pour commencer l'analyse.")
    st.stop()


# 4. SIDEBAR - PARAMÈTRES
st.sidebar.title("⚙️ Paramètres")

min_d, max_d = raw_df['startdate'].min().date(), raw_df['startdate'].max().date()
dates = st.sidebar.date_input("Période d'analyse :", value=(min_d, max_d), min_value=min_d, max_value=max_d)


st.sidebar.caption("🎧 Données acoustiques : espèces considérées comme sauvages par défaut")
filtre_sauvage = "Toutes"

st.sidebar.subheader("🎧 Filtre qualité BirdNET")

seuil_confiance = st.sidebar.slider(
    "Indice de confiance minimum :",
    min_value=0.0,
    max_value=1.0,
    value=0.8,   # valeur par défaut recommandée
    step=0.05
)

# 5. APPLICATION DES FILTRES
df = raw_df.copy()

# Filtre sur la confiance BirdNET (si disponible)
if 'Birdnet_confidence_index' in df.columns:
    df['Birdnet_confidence_index'] = pd.to_numeric(df['Birdnet_confidence_index'], errors='coerce')
    df = df[df['Birdnet_confidence_index'] >= seuil_confiance]

if isinstance(dates, tuple) and len(dates) == 2:
    df = df[(df['startdate'].dt.date >= dates[0]) & (df['startdate'].dt.date <= dates[1])]

df_base_date = df.copy()

st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Repérage spatial")

with st.sidebar:
    show_sites_map_popover(
        df_base_date,
        label="📍 Voir la carte de repérage",
        zoom=14
    )

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Focus Espèce")

liste_especes = sorted(df_base_date['vernacular_name'].dropna().unique().tolist())
liste_especes.insert(0, "Toutes les espèces")

espece_sidebar = st.sidebar.selectbox(
    "Rechercher une espèce :",
    options=liste_especes,
    index=0
)

if espece_sidebar != "Toutes les espèces":
    df = df[df['vernacular_name'] == espece_sidebar]


# 6. FONCTIONS BOOTSTRAP
def bootstrap_stats(data, n_samples=1000, n_iterations=400):
    if len(data) < 10:
        return None

    res = {'S': [], 'H': [], 'InvD': [], 'J': []}

    for _ in range(n_iterations):
        sample = data.sample(n=n_samples, replace=True)

        # Si detection_count existe, on l'utilise.
        # Sinon, chaque ligne compte pour 1 observation.
        if 'detection_count' in sample.columns:
            abundance = sample.groupby('vernacular_name')['detection_count'].sum()
        else:
            abundance = sample.groupby('vernacular_name').size()

        pi = abundance / abundance.sum()
        s_v = len(abundance)
        h_v = -1 * (pi * np.log(pi + 1e-9)).sum()
        inv_d_v = 1 / (pi ** 2).sum()
        res['S'].append(s_v)
        res['H'].append(h_v)
        res['InvD'].append(inv_d_v)
        res['J'].append(h_v / np.log(s_v) if s_v > 1 else 0)

    return {k: (np.mean(v), np.std(v)) for k, v in res.items()}

def detect_effort_unit_column(data):
    """
    Détecte automatiquement la meilleure colonne à utiliser
    comme unité matérielle d'échantillonnage.

    Priorité proposée :
    1. Sensor          -> vrai identifiant de capteur/caméra si disponible
    2. sensor
    3. camera_id
    4. camera
    5. site            -> repli par défaut si on n'a pas mieux

    Returns
    -------
    str
        Nom de la colonne retenue
    """
    candidates = ['Sensor', 'sensor', 'camera_id', 'camera', 'site']

    for col in candidates:
        if col in data.columns:
            return col

    # Si aucune colonne n'existe, on retourne None
    return None


def compute_sampling_effort(data, effort_col=None):
    """
    Calcule un effort d'échantillonnage robuste à partir
    des jours réellement actifs dans le dataset.

    Logique :
    - on ne suppose PAS que chaque caméra a tourné tous les jours
    - on compte uniquement les combinaisons réellement observées :
      (unité d'effort, date)

    Exemple :
    - caméra A active 10 jours
    - caméra B active 7 jours
    => effort total = 17 camera-jours

    Si aucune colonne caméra explicite n'existe,
    on retombe sur 'site' comme proxy.

    Parameters
    ----------
    data : pd.DataFrame
        Jeu de données filtré
    effort_col : str or None
        Colonne à utiliser comme unité d'effort.
        Si None, détection automatique.

    Returns
    -------
    effort : int
        Nombre total d'unités d'effort actives
    effort_col : str or None
        Colonne effectivement utilisée
    effort_table : pd.DataFrame
        Tableau intermédiaire des unités actives par date
    """
    if data.empty or 'startdate' not in data.columns:
        return 0, effort_col, pd.DataFrame()

    # Détection automatique de la colonne d'effort si non fournie
    if effort_col is None:
        effort_col = detect_effort_unit_column(data)

    # Si aucune colonne utilisable n'existe, effort non calculable
    if effort_col is None or effort_col not in data.columns:
        return 0, effort_col, pd.DataFrame()

    # On extrait la date calendaire
    df_effort = data.copy()
    df_effort['date_only'] = pd.to_datetime(df_effort['startdate']).dt.date

    # On garde uniquement les couples uniques (capteur/jour)
    effort_table = df_effort[[effort_col, 'date_only']].dropna().drop_duplicates()

    # L'effort total correspond au nombre de lignes uniques
    effort = len(effort_table)

    return effort, effort_col, effort_table

def bootstrap_iajc(data, n_iterations=400, effort_col=None):
    """
    IAJC = nombre total de détections / effort réel d'échantillonnage

    effort réel = nombre de couples uniques (capteur, jour)
    ou à défaut (site, jour)
    """
    if data.empty:
        return 0.0, 0.0

    effort, effort_col_used, effort_table = compute_sampling_effort(data, effort_col=effort_col)

    if effort <= 0:
        return 0.0, 0.0

    iajc_sims = []

    for _ in range(n_iterations):
        sample = data.sample(frac=1.0, replace=True)
        total_animaux = sample['detection_count'].sum()
        iajc_sims.append(total_animaux / effort)

    return float(np.mean(iajc_sims)), float(np.std(iajc_sims))


# 7. FONCTIONS STATS / VISU
def plot_matrice_jaccard(df_input):
    import plotly.graph_objects as go

    sites = sorted(df_input['site'].unique())
    species_per_site = {
        site: set(df_input[df_input['site'] == site]['vernacular_name'].unique())
        for site in sites
    }

    n = len(sites)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            s1, s2 = species_per_site[sites[i]], species_per_site[sites[j]]
            intersection = len(s1.intersection(s2))
            union = len(s1.union(s2))
            matrix[i, j] = intersection / union if union > 0 else 0

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=sites,
        y=sites,
        colorscale=[[0, "#FFFFFF"], [0.5, C_JAUNE], [1, C_VERT_SOMBRE]],
        text=np.round(matrix, 2),
        texttemplate="<b>%{text}</b>",
        showscale=True
    ))

    fig.update_layout(
        title="🤝 Similitude de Jaccard (Composition des espèces)",
        paper_bgcolor=C_FOND,
        plot_bgcolor=C_FOND,
        height=500,
        margin=dict(t=50, b=50),
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed")
    )
    return fig


def plot_courbe_accumulation(df_input):
    import plotly.graph_objects as go

    df_chrono = df_input.sort_values('startdate').copy()
    df_chrono['date_uniquement'] = df_chrono['startdate'].dt.date

    first_seen = df_chrono.groupby('vernacular_name')['date_uniquement'].min().reset_index()
    daily_new = first_seen.groupby('date_uniquement').size().reset_index(name='nouveaux')

    toutes_dates = pd.DataFrame({
        'date_uniquement': pd.date_range(
            df_chrono['date_uniquement'].min(),
            df_chrono['date_uniquement'].max()
        ).date
    })

    df_final = pd.merge(toutes_dates, daily_new, on='date_uniquement', how='left').fillna(0)
    df_final['cumul'] = df_final['nouveaux'].cumsum()

    counts = df_input['vernacular_name'].value_counts()
    s_obs = len(counts)
    f1 = sum(counts == 1)
    f2 = sum(counts == 2)

    if f2 > 0:
        chao1 = s_obs + (f1 ** 2 / (2 * f2))
    else:
        chao1 = s_obs + (f1 * (f1 - 1)) / 2

    completude = (s_obs / chao1) * 100 if chao1 > 0 else 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_final['date_uniquement'],
        y=df_final['cumul'],
        mode='lines+markers',
        name=f"Observé ({s_obs})",
        line=dict(color=C_VERT_SOMBRE, width=3),
        fill='tozeroy',
        fillcolor='rgba(45, 78, 40, 0.1)'
    ))

    fig.add_trace(go.Scatter(
        x=[df_final['date_uniquement'].min(), df_final['date_uniquement'].max()],
        y=[chao1, chao1],
        mode='lines',
        name=f"Potentiel estimé ({chao1:.1f})",
        line=dict(color=C_ROUGE, width=2, dash='dash')
    ))

    fig.update_layout(
        title=f"📈 Taux de complétude de l'inventaire : {completude:.1f}%",
        xaxis_title="Calendrier du suivi",
        yaxis_title="Nombre d'espèces",
        paper_bgcolor=C_FOND,
        plot_bgcolor=C_FOND,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450
    )

    return fig, completude


def plot_dendrogramme_jaccard(df_input):
    """
    Construit un dendrogramme basé sur une vraie distance de Jaccard.

    Principe :
    - on transforme les données en matrice binaire présence/absence
      des espèces par site
    - on calcule la distance de Jaccard entre les sites
    - on applique un clustering hiérarchique
    - on affiche le dendrogramme correspondant

    Important :
    - le titre 'Jaccard' est alors méthodologiquement correct
    - plus deux sites fusionnent bas dans l'arbre,
      plus leur composition spécifique est proche
    """

    import plotly.figure_factory as ff
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage

    # ---------------------------------------------------------
    # 1. Vérifications minimales
    # ---------------------------------------------------------
    # On vérifie que les colonnes nécessaires existent bien.
    # Sans cela, impossible de construire une matrice espèces x sites.
    # ---------------------------------------------------------
    required_cols = {'site', 'vernacular_name'}
    if not required_cols.issubset(df_input.columns):
        fig = ff.create_dendrogram([[0], [1]], labels=["Erreur", "Colonnes manquantes"])
        fig.update_layout(
            title="Colonnes nécessaires absentes : 'site' et/ou 'vernacular_name'",
            paper_bgcolor=C_FOND,
            plot_bgcolor=C_FOND
        )
        return fig

    # ---------------------------------------------------------
    # 2. Construction de la matrice présence / absence
    # ---------------------------------------------------------
    # On crée un tableau :
    # - lignes   = espèces
    # - colonnes = sites
    # - valeurs  = 1 si l'espèce est présente, 0 sinon
    #
    # Ensuite on transpose pour obtenir :
    # - lignes = sites
    # - colonnes = espèces
    #
    # C'est ce format qu'on utilise pour calculer les distances entre sites.
    # ---------------------------------------------------------
    pivot = df_input.pivot_table(
        index='vernacular_name',
        columns='site',
        aggfunc='size',
        fill_value=0
    )

    # Transformation en binaire présence / absence
    pivot = (pivot > 0).astype(int).T

    # ---------------------------------------------------------
    # 3. Garde-fou : il faut au moins 2 sites
    # ---------------------------------------------------------
    if pivot.shape[0] < 2:
        fig = ff.create_dendrogram([[0], [1]], labels=["Pas assez", "de sites"])
        fig.update_layout(
            title="Le dendrogramme nécessite au moins 2 sites",
            paper_bgcolor=C_FOND,
            plot_bgcolor=C_FOND
        )
        return fig

    # ---------------------------------------------------------
    # 4. Calcul de la vraie distance de Jaccard
    # ---------------------------------------------------------
    # pdist(..., metric='jaccard') calcule les distances entre toutes
    # les paires de sites à partir de la matrice binaire.
    #
    # La distance de Jaccard vaut :
    # 0   = composition identique
    # 1   = aucune espèce en commun
    # ---------------------------------------------------------
    dist_jaccard = pdist(pivot.values, metric='jaccard')

    # ---------------------------------------------------------
    # 5. Clustering hiérarchique
    # ---------------------------------------------------------
    # On construit l'arbre hiérarchique à partir de la distance de Jaccard.
    #
    # Méthode recommandée ici :
    # - 'average' (UPGMA)
    #
    # Pourquoi pas 'ward' ?
    # - parce que Ward n'est pas adapté aux distances de Jaccard
    # - Ward suppose une logique euclidienne / variance
    # ---------------------------------------------------------
    linkage_matrix = linkage(dist_jaccard, method='average')

    # ---------------------------------------------------------
    # 6. Création du dendrogramme Plotly
    # ---------------------------------------------------------
    # Ici, on passe explicitement :
    # - la matrice des sites x espèces
    # - la fonction linkage maison
    #
    # Ainsi, le dendrogramme repose bien sur notre distance de Jaccard.
    # ---------------------------------------------------------
    fig = ff.create_dendrogram(
        X=pivot.values,
        labels=pivot.index.tolist(),
        linkagefun=lambda _: linkage_matrix,
        color_threshold=None
    )

    # ---------------------------------------------------------
    # 7. Mise en forme graphique
    # ---------------------------------------------------------
    # On harmonise avec ta charte et on renomme correctement l'axe Y.
    # Ici, l'axe Y représente une DISTANCE écologique de Jaccard.
    # ---------------------------------------------------------
    fig.update_layout(
        title="🌳 Dendrogramme de proximité biologique (distance de Jaccard)",
        xaxis_title="Sites",
        yaxis_title="Distance écologique (Jaccard)",
        paper_bgcolor=C_FOND,
        plot_bgcolor=C_FOND,
        height=450,
        margin=dict(l=50, r=100, t=80, b=120),
        xaxis=dict(
            tickangle=-45,
            automargin=True,
            showgrid=False
        )
    )

    return fig

def build_nonsignificance_matrix(groups, sig_pairs):
    """
    Construit une matrice booléenne de non-significativité entre groupes.

    Principe :
    - True  = les 2 groupes NE sont PAS significativement différents
    - False = les 2 groupes sont significativement différents

    Parameters
    ----------
    groups : list[str]
        Liste ordonnée des groupes (sites).
    sig_pairs : set[tuple[str, str]]
        Ensemble des paires significativement différentes.
        Chaque paire doit être stockée sous forme triée :
        tuple(sorted((g1, g2)))

    Returns
    -------
    pd.DataFrame
        Matrice carrée booléenne indexée par les noms de groupes.
    """
    # On initialise une matrice True partout :
    # par défaut, on considère que les groupes ne sont pas significativement différents
    mat = pd.DataFrame(True, index=groups, columns=groups)

    # On remplit ensuite chaque case
    for g1 in groups:
        for g2 in groups:
            if g1 == g2:
                # Un groupe comparé à lui-même est forcément non significatif
                mat.loc[g1, g2] = True
            else:
                # On normalise la paire pour éviter les doublons (A,B) / (B,A)
                pair = tuple(sorted((g1, g2)))

                # Si la paire est dans sig_pairs => différence significative => False
                # Sinon => non significatif => True
                mat.loc[g1, g2] = pair not in sig_pairs

    return mat


def compact_letter_display_from_nonsig(groups, nonsig_matrix):
    """
    Génère un vrai Compact Letter Display (CLD) avec chevauchements
    de type A / AB / BC / C.

    Règles à respecter :
    1. Si deux groupes partagent une lettre, ils ne doivent PAS être significativement différents.
    2. Si deux groupes ne sont PAS significativement différents, ils doivent partager au moins une lettre.

    Exemple attendu :
    - Site1 = A
    - Site2 = AB
    - Site3 = BC
    - Site4 = C

    Parameters
    ----------
    groups : list[str]
        Groupes ordonnés comme on veut les afficher
        (en général par moyenne décroissante).
    nonsig_matrix : pd.DataFrame
        Matrice booléenne de non-significativité.

    Returns
    -------
    dict[str, str]
        Dictionnaire du type :
        {
            "Site 1": "A",
            "Site 2": "AB",
            "Site 3": "BC"
        }
    """

    # Dictionnaire final : pour chaque groupe, liste des lettres attribuées
    letters_sets = {g: [] for g in groups}

    # Chaque "colonne" représente une lettre potentielle.
    # Exemple :
    # [
    #   ['Site1', 'Site2'],   -> lettre A
    #   ['Site2', 'Site3'],   -> lettre B
    #   ['Site3', 'Site4']    -> lettre C
    # ]
    letter_columns = []

    # ---------------------------------------------------------
    # ÉTAPE 1 : placement glouton
    # ---------------------------------------------------------
    # On parcourt les groupes dans l'ordre demandé.
    # Pour chaque groupe, on essaie de l'ajouter aux colonnes existantes
    # si cela reste compatible avec tous les groupes déjà présents.
    # Compatible = non significatif avec tous les groupes de la colonne.
    # ---------------------------------------------------------
    for g in groups:
        placed_in_existing = []

        for i, col_groups in enumerate(letter_columns):
            # Le groupe g peut-il rejoindre cette colonne ?
            compatible = all(nonsig_matrix.loc[g, other] for other in col_groups)

            if compatible:
                col_groups.append(g)
                placed_in_existing.append(i)

        # Si aucune colonne existante n'est compatible,
        # on crée une nouvelle colonne rien que pour lui.
        if not placed_in_existing:
            letter_columns.append([g])

    # ---------------------------------------------------------
    # ÉTAPE 2 : vérification de couverture
    # ---------------------------------------------------------
    # Une méthode gloutonne simple ne garantit pas toujours que
    # toutes les paires NON significatives partagent une lettre.
    #
    # Donc on vérifie explicitement cette propriété.
    # Si une paire non significative ne partage aucune lettre,
    # on crée une nouvelle colonne qui les contient.
    # ---------------------------------------------------------
    changed = True
    while changed:
        changed = False

        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                # On ne s'intéresse qu'aux paires NON significatives
                if nonsig_matrix.loc[g1, g2]:

                    # Vérifie s'ils partagent déjà une colonne / lettre
                    share_letter = any(
                        (g1 in col and g2 in col)
                        for col in letter_columns
                    )

                    # Si ce n'est pas le cas, on crée une nouvelle colonne
                    if not share_letter:
                        new_col = [g1, g2]

                        # On essaie d'élargir cette nouvelle colonne
                        # à d'autres groupes compatibles avec TOUS les membres
                        for g3 in groups:
                            if g3 not in new_col:
                                if all(nonsig_matrix.loc[g3, x] for x in new_col):
                                    new_col.append(g3)

                        letter_columns.append(new_col)
                        changed = True

    # ---------------------------------------------------------
    # ÉTAPE 3 : suppression des colonnes redondantes
    # ---------------------------------------------------------
    # Si une colonne est strictement incluse dans une autre,
    # elle n'apporte aucune information supplémentaire.
    # On la supprime pour simplifier le résultat final.
    # ---------------------------------------------------------
    cleaned_columns = []

    for i, col_i in enumerate(letter_columns):
        set_i = set(col_i)
        redundant = False

        for j, col_j in enumerate(letter_columns):
            if i != j:
                set_j = set(col_j)

                # Si col_i est incluse dans col_j, et pas égale,
                # alors col_i est redondante
                if set_i.issubset(set_j) and set_i != set_j:
                    redundant = True
                    break

        if not redundant:
            cleaned_columns.append(col_i)

    # ---------------------------------------------------------
    # ÉTAPE 4 : transformation des colonnes en lettres
    # ---------------------------------------------------------
    # 1ère colonne -> A
    # 2ème colonne -> B
    # 3ème colonne -> C
    # etc.
    # ---------------------------------------------------------
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # Sécurité si jamais il y a énormément de colonnes
    if len(cleaned_columns) > len(alphabet):
        alphabet.extend([f"A{i}" for i in range(1, 100)])

    for letter, col_groups in zip(alphabet, cleaned_columns):
        for g in col_groups:
            letters_sets[g].append(letter)

    # On concatène les lettres de chaque groupe
    return {
        g: "".join(letters_sets[g]) if letters_sets[g] else ""
        for g in groups
    }

def plot_tukey_shannon(df_input):
    """
    Calcule l'indice de Shannon par site et par semaine,
    applique un test post-hoc de Tukey HSD,
    puis affiche un graphique en barres des moyennes
    avec erreurs standard et lettres de groupes significatifs
    de type A / AB / BC / C.

    Important :
    - les sites sont triés par moyenne décroissante
    - les lettres reflètent les similarités statistiques réelles
    """

    import plotly.graph_objects as go


     # ---------------------------------------------------------
    # 1. Préparation des données
    # ---------------------------------------------------------
    # On n'utilise plus Annee + Semaine ISO séparées,
    # mais l'identifiant week_id créé dans load_data(),
    # ce qui garantit une définition homogène
    # de la semaine dans tout le script.
    # ---------------------------------------------------------
    df_temp = df_input.copy()

    # ---------------------------------------------------------
    # 2. Fonction locale pour calculer Shannon
    # ---------------------------------------------------------
    def calc_shannon(counts):
        """
        Calcule l'indice de Shannon à partir d'une série d'abondances.
        """
        probs = counts / (counts.sum() + 1e-9)
        return -np.sum(probs * np.log(probs + 1e-9))

    # ---------------------------------------------------------
    # 3. Calcul du Shannon hebdomadaire par site
    # ---------------------------------------------------------
    # On agrège d'abord les détections par espèce,
    # puis on calcule Shannon à l'échelle (site, week_id).
    # ---------------------------------------------------------
    df_stats = df_temp.groupby(['site', 'week_id']).apply(
        lambda x: calc_shannon(x.groupby('vernacular_name')['detection_count'].sum())
    ).reset_index(name='shannon')

    # ---------------------------------------------------------
    # 4. Garde-fou : on conserve seulement les sites ayant
    # au moins 2 valeurs hebdomadaires
    # ---------------------------------------------------------
    counts_per_site = df_stats.groupby('site').size()
    valid_sites = counts_per_site[counts_per_site >= 2].index.tolist()
    df_stats = df_stats[df_stats['site'].isin(valid_sites)].copy()

    # Si moins de 2 sites valides, on ne peut pas faire Tukey
    if df_stats['site'].nunique() < 2:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=C_FOND,
            plot_bgcolor=C_FOND,
            title="Pas assez de données pour Tukey HSD"
        )
        return fig

    # ---------------------------------------------------------
    # 5. Résumé statistique par site
    # ---------------------------------------------------------
    # mean = moyenne de Shannon hebdomadaire
    # std  = écart-type
    # count = nombre de semaines
    # sem = erreur standard de la moyenne
    # ---------------------------------------------------------
    stats_summary = df_stats.groupby('site')['shannon'].agg(['mean', 'std', 'count']).reset_index()
    stats_summary['sem'] = stats_summary['std'] / np.sqrt(stats_summary['count'])

    # Tri décroissant par moyenne pour un affichage plus lisible
    stats_summary = stats_summary.sort_values('mean', ascending=False)
    ordered_sites = stats_summary['site'].tolist()

    # ---------------------------------------------------------
    # 6. Test post-hoc de Tukey HSD
    # ---------------------------------------------------------
    res = pairwise_tukeyhsd(df_stats['shannon'], df_stats['site'])

    # Conversion du tableau de résultats en DataFrame pandas
    results_df = pd.DataFrame(
        data=res.summary().data[1:],
        columns=res.summary().data[0]
    )

    # ---------------------------------------------------------
    # 7. Construction de la liste des paires significatives
    # ---------------------------------------------------------
    # On stocke uniquement les comparaisons rejetées
    # sous forme de tuples triés :
    # ('Site A', 'Site B')
    # ---------------------------------------------------------
    sig_pairs = set()

    for _, row in results_df.iterrows():
        g1 = str(row['group1'])
        g2 = str(row['group2'])
        reject = bool(row['reject'])

        if reject:
            sig_pairs.add(tuple(sorted((g1, g2))))

    # ---------------------------------------------------------
    # 8. Construction de la matrice de non-significativité
    # puis des lettres compactes
    # ---------------------------------------------------------
    nonsig_matrix = build_nonsignificance_matrix(ordered_sites, sig_pairs)
    letters = compact_letter_display_from_nonsig(ordered_sites, nonsig_matrix)

    # ---------------------------------------------------------
    # 9. Construction du graphique en barres
    # ---------------------------------------------------------
    fig = go.Figure(go.Bar(
        x=stats_summary['site'],
        y=stats_summary['mean'],

        # Barres d'erreur = SEM
        error_y=dict(
            type='data',
            array=stats_summary['sem'],
            visible=True,
            color=C_JAUNE
        ),

        marker_color=C_VERT_SOMBRE,

        # Affichage des lettres au-dessus des barres
        text=[f"<b>{letters[s]}</b>" for s in stats_summary['site']],
        textposition='outside',
        textfont=dict(size=18, color=C_ROUGE)
    ))

    # Hauteur max pour laisser de la place aux lettres
    y_max = (stats_summary['mean'] + stats_summary['sem']).max()

    fig.update_layout(
        paper_bgcolor=C_FOND,
        plot_bgcolor=C_FOND,
        xaxis_title="Site",
        yaxis=dict(
            title="Shannon (H') moyen / semaine",
            range=[0, y_max * 1.25]
        )
    )

    return fig


def plot_kruskal_shannon(df_input):
    """
    Calcule l'indice de Shannon par site et par semaine,
    applique un test global de Kruskal-Wallis,
    puis un post-hoc de Dunn si le test global est significatif.

    Affiche ensuite un boxplot avec lettres de groupes significatifs
    de type A / AB / BC / C.

    Returns
    -------
    p_global : float
        p-value du test global de Kruskal-Wallis
    letters : dict
        Lettres attribuées à chaque site
    fig : plotly figure
        Figure finale
    """

    # ---------------------------------------------------------
    # 1. Préparation temporelle
    # ---------------------------------------------------------
    # On s'aligne sur week_id, identifiant hebdomadaire
    # standardisé créé dès le chargement des données.
    # ---------------------------------------------------------
    df_temp = df_input.copy()

    # ---------------------------------------------------------
    # 2. Fonction locale de calcul de Shannon
    # ---------------------------------------------------------
    # Ici on travaille bien sur les abondances par espèce
    # et non sur un simple value_counts() brut.
    # ---------------------------------------------------------
    def calc_shannon_from_counts(group):
        counts = group.groupby('vernacular_name')['detection_count'].sum()
        probs = counts / (counts.sum() + 1e-9)
        return -np.sum(probs * np.log(probs + 1e-9))

    # ---------------------------------------------------------
    # 3. Calcul du Shannon par site et par semaine
    # ---------------------------------------------------------
    df_stats = df_temp.groupby(['site', 'week_id']).apply(
        calc_shannon_from_counts
    ).reset_index(name='shannon')

    # ---------------------------------------------------------
    # 3. Garde-fou : au moins 2 semaines par site
    # ---------------------------------------------------------
    counts_per_site = df_stats.groupby('site').size()
    valid_sites = counts_per_site[counts_per_site >= 2].index.tolist()
    df_stats = df_stats[df_stats['site'].isin(valid_sites)].copy()

    # Si moins de 2 sites valides, impossible de comparer
    if df_stats['site'].nunique() < 2:
        fig = px.box()
        fig.update_layout(
            paper_bgcolor=C_FOND,
            plot_bgcolor=C_FOND,
            title="Pas assez de données pour Kruskal-Wallis"
        )
        return 1.0, {}, fig

    # ---------------------------------------------------------
    # 4. Ordre des sites : médiane décroissante
    # ---------------------------------------------------------
    order = df_stats.groupby('site')['shannon'].median().sort_values(ascending=False).index.tolist()

    # ---------------------------------------------------------
    # 5. Test global de Kruskal-Wallis
    # ---------------------------------------------------------
    stat_kw, p_global = scipy_stats.kruskal(
        *[g['shannon'].values for _, g in df_stats.groupby('site')]
    )

    # ---------------------------------------------------------
    # 6. Si le test global est significatif :
    #    on applique un post-hoc de Dunn avec correction Bonferroni
    # ---------------------------------------------------------
    if p_global < 0.05:
        p_matrix = sp.posthoc_dunn(
            df_stats,
            val_col='shannon',
            group_col='site',
            p_adjust='bonferroni'
        )

        # Construction des paires significatives
        sig_pairs = set()

        for i, g1 in enumerate(order):
            for g2 in order[i + 1:]:
                pval = p_matrix.loc[g1, g2]

                if pval < 0.05:
                    sig_pairs.add(tuple(sorted((g1, g2))))

        # Matrice de non-significativité
        nonsig_matrix = build_nonsignificance_matrix(order, sig_pairs)

        # Lettres compactes
        letters = compact_letter_display_from_nonsig(order, nonsig_matrix)

    else:
        # Si le test global n'est pas significatif,
        # tout le monde partage la même lettre
        letters = {site: "A" for site in order}

    # ---------------------------------------------------------
    # 7. Construction du boxplot
    # ---------------------------------------------------------
    fig = px.box(
        df_stats,
        x='site',
        y='shannon',
        color='site',
        category_orders={"site": order},
        points="all"
    )

    fig.update_traces(marker_color=C_VERT_SOMBRE)

    # ---------------------------------------------------------
    # 8. Ajout des lettres au-dessus des boîtes
    # ---------------------------------------------------------
    for site in order:
        y_max = df_stats[df_stats['site'] == site]['shannon'].max()

        fig.add_annotation(
            x=site,
            y=y_max * 1.12,
            text=f"<b>{letters[site]}</b>",
            showarrow=False,
            font=dict(color=C_ROUGE, size=18)
        )

    fig.update_layout(
        paper_bgcolor=C_FOND,
        plot_bgcolor=C_FOND,
        showlegend=False,
        yaxis_title="Shannon (H') par semaine"
    )

    return p_global, letters, fig


def plot_activity_heatmap(df_input):
    import pandas as pd

    df_temp = df_input.copy()
    if 'Heure' not in df_temp.columns:
        df_temp['Heure'] = df_temp['startdate'].dt.hour

    activity = df_temp.groupby(['site', 'Heure']).size().reset_index(name='count')
    pivot_activity = activity.pivot(index='site', columns='Heure', values='count').fillna(0)

    for hour in range(24):
        if hour not in pivot_activity.columns:
            pivot_activity[hour] = 0

    pivot_activity = pivot_activity.reindex(columns=sorted(pivot_activity.columns))
    pivot_norm = pivot_activity.div(pivot_activity.max(axis=1) + 1e-9, axis=0)

    x_labels = [f"{int(h)}h" for h in pivot_norm.columns]

    fig = px.imshow(
        pivot_norm,
        labels=dict(x="Heure de la journée", y="Sites", color="Intensité"),
        x=x_labels,
        color_continuous_scale='Viridis',
        aspect="auto"
    )

    fig.update_layout(
        title="🕒 Profil d'activité nycthémérale (Pics normalisés)",
        paper_bgcolor=C_FOND,
        plot_bgcolor=C_FOND,
        height=450
    )

    return fig


def plot_indicator_species(df_input):
    df_temp = df_input.copy()
    df_temp['Semaine'] = df_temp['startdate'].dt.isocalendar().week

    abundance = df_temp.groupby(['site', 'Semaine', 'vernacular_name']).size().reset_index(name='count')
    matrix = abundance.groupby(['site', 'vernacular_name'])['count'].mean().unstack(fill_value=0)

    A = matrix.div(matrix.sum(axis=0), axis=1)
    presence = df_temp.groupby(['site', 'Semaine', 'vernacular_name']).size().unstack(fill_value=0) > 0
    B = presence.groupby(level='site').mean()
    indval = A * B

    best_indicators = []
    for site in indval.index:
        row = indval.loc[site]
        best_sp = row.idxmax()
        score = row.max()
        if score > 0:
            best_indicators.append({'site': site, 'espece': best_sp, 'score': score})

    df_res = pd.DataFrame(best_indicators).sort_values('score', ascending=True)

    palette_bio = [
        C_VERT_SOMBRE, C_JAUNE, C_ROUGE, C_BLEU,
        C_VIOLET, C_ROSE, C_VERT_CLAIR, C_MAUVE
    ]

    fig = px.bar(
        df_res,
        x='score',
        y='site',
        color='espece',
        orientation='h',
        text='espece',
        title="✨ Signature Biologique (Espèces Indicatrices) par Site",
        labels={'score': 'Indice de Spécificité', 'site': 'Site'},
        color_discrete_sequence=palette_bio
    )

    fig.update_layout(
        paper_bgcolor=C_FOND,
        plot_bgcolor=C_FOND,
        showlegend=False,
        height=400,
        font=dict(color=C_VERT_SOMBRE),
        xaxis=dict(
            title="Force de l'indicateur (0 à 1)",
            gridcolor=C_GRIS,
            range=[0, df_res['score'].max() * 1.2 if not df_res.empty else 1]
        ),
        yaxis=dict(gridcolor=C_GRIS)
    )

    fig.update_traces(
        textposition='inside',
        textfont=dict(size=14, color='white'),
        marker_line_color=C_VERT_SOMBRE,
        marker_line_width=1
    )

    return fig


def plot_diurne_nocturne(df_input):
    df_temp = df_input.copy()
    df_temp['Periode'] = df_temp['Heure'].apply(lambda x: 'Jour ☀️' if 7 <= x < 19 else 'Nuit 🌙')

    df_dist = df_temp.groupby(['site', 'Periode']).size().reset_index(name='count')
    total_site = df_dist.groupby('site')['count'].transform('sum')
    df_dist['Pourcentage'] = (df_dist['count'] / total_site) * 100

    nocturnite = df_dist[df_dist['Periode'] == 'Nuit 🌙'].sort_values('Pourcentage', ascending=False)
    order = nocturnite['site'].tolist()

    fig = px.bar(
        df_dist,
        x='site',
        y='Pourcentage',
        color='Periode',
        text=df_dist['Pourcentage'].apply(lambda x: f"{x:.1f}%"),
        category_orders={'site': order},
        color_discrete_map={'Jour ☀️': C_JAUNE, 'Nuit 🌙': C_VIOLET},
        title="🌓 Répartition de l'Activité (Jour vs Nuit)",
        barmode='stack'
    )

    fig.update_layout(
        paper_bgcolor=C_FOND,
        plot_bgcolor=C_FOND,
        yaxis_title="% de détections",
        xaxis_title=None,
        legend_title=None,
        height=450
    )

    fig.update_traces(textposition='inside', textfont=dict(color='white'))

    return fig

# =========================================================
# CALCUL SHANNON HEBDOMADAIRE (UTILISÉ POUR LE DIAGNOSTIC)
# =========================================================
@st.cache_data
def compute_weekly_shannon_distribution(df_input):
    """
    Calcule la distribution hebdomadaire de l'indice de Shannon.

    Cette fonction sert notamment à :
    - mesurer la stabilité temporelle
    - alimenter le diagnostic écologique
    - éviter de recalculer plusieurs fois la même logique
      dans différents onglets

    Principe :
    1. on agrège les détections par site / semaine / espèce
    2. on calcule Shannon pour chaque couple (site, semaine)
    """
    # ---------------------------------------------------------
    # 1. Garde-fou : dataset vide
    # ---------------------------------------------------------
    if df_input.empty:
        return pd.DataFrame(columns=['site', 'week_id', 'shannon_val'])

    # ---------------------------------------------------------
    # 2. Agrégation hebdomadaire des abondances par espèce
    # ---------------------------------------------------------
    # On utilise week_id défini dans load_data()
    # pour garantir une logique temporelle cohérente partout.
    # ---------------------------------------------------------
    df_weekly_stats = df_input.groupby(
        ['site', 'week_id', 'vernacular_name']
    )['detection_count'].sum().reset_index()

    # ---------------------------------------------------------
    # 3. Fonction locale de calcul de Shannon
    # ---------------------------------------------------------
    def compute_shannon(group):
        total = group['detection_count'].sum()

        if total == 0:
            return 0.0

        p_i = group['detection_count'] / total
        return float(-1 * (p_i * np.log(p_i + 1e-9)).sum())

    # ---------------------------------------------------------
    # 4. Calcul du Shannon par site et par semaine
    # ---------------------------------------------------------
    df_dist_shannon = df_weekly_stats.groupby(
        ['site', 'week_id']
    ).apply(
        compute_shannon,
        include_groups=False
    ).reset_index(name='shannon_val')

    return df_dist_shannon

@st.cache_data
def prepare_long_term_indicators(df_input, grain="M"):
    """
    Calcule les indicateurs de biodiversité à un pas de temps long terme.

    Parameters
    ----------
    df_input : pd.DataFrame
        Jeu de données filtré.
    grain : str
        Grain temporel :
        - "M" = mois
        - "Q" = trimestre
        - "Y" = année

    Returns
    -------
    pd.DataFrame
        Tableau avec une ligne par site et par période.
    """
    # ---------------------------------------------------------
    # 1. Garde-fou : dataset vide ou absent
    # ---------------------------------------------------------
    empty_cols = [
        "site",
        "period_label",
        "period_start",
        "richesse",
        "shannon",
        "pielou",
        "simpson_inv_d",
        "detections",
        "events"
    ]

    if df_input is None or df_input.empty:
        return pd.DataFrame(columns=empty_cols)

    df_temp = df_input.copy()

    # ---------------------------------------------------------
    # 2. Vérification minimale des colonnes utiles
    # ---------------------------------------------------------
    required_cols = {"site", "startdate", "vernacular_name", "detection_count"}
    missing_cols = required_cols - set(df_temp.columns)

    if missing_cols:
        return pd.DataFrame(columns=empty_cols)

    # ---------------------------------------------------------
    # 3. Sécurisation des types
    # ---------------------------------------------------------
    df_temp["startdate"] = pd.to_datetime(df_temp["startdate"], errors="coerce")
    df_temp["detection_count"] = pd.to_numeric(df_temp["detection_count"], errors="coerce").fillna(1)

    df_temp = df_temp.dropna(subset=["site", "startdate", "vernacular_name"])

    if df_temp.empty:
        return pd.DataFrame(columns=empty_cols)

    # ---------------------------------------------------------
    # 4. Construction de la période selon le grain choisi
    # ---------------------------------------------------------
    if grain == "M":
        df_temp["period"] = df_temp["startdate"].dt.to_period("M")
    elif grain == "Q":
        df_temp["period"] = df_temp["startdate"].dt.to_period("Q")
    elif grain == "Y":
        df_temp["period"] = df_temp["startdate"].dt.to_period("Y")
    else:
        raise ValueError("grain doit valoir 'M', 'Q' ou 'Y'")

    df_temp["period_start"] = df_temp["period"].dt.start_time
    df_temp["period_label"] = df_temp["period"].astype(str)

    # ---------------------------------------------------------
    # 5. Calcul des indicateurs par site et période
    # ---------------------------------------------------------
    results = []

    grouped = df_temp.groupby(["site", "period_label", "period_start"], dropna=False)

    for (site, period_label, period_start), group in grouped:
        abundance = group.groupby("vernacular_name")["detection_count"].sum()
        total = abundance.sum()

        if total <= 0 or len(abundance) == 0:
            richesse = 0
            shannon = 0.0
            pielou = 0.0
            simpson_inv_d = 0.0
            detections = 0.0
            events = int(len(group))
        else:
            pi = abundance / total
            richesse = int(len(abundance))
            shannon = float(-1 * (pi * np.log(pi + 1e-9)).sum())
            simpson_inv_d = float(1 / (pi ** 2).sum())
            pielou = float(shannon / np.log(richesse) if richesse > 1 else 0.0)
            detections = float(total)
            events = int(len(group))

        results.append({
            "site": site,
            "period_label": period_label,
            "period_start": period_start,
            "richesse": richesse,
            "shannon": shannon,
            "pielou": pielou,
            "simpson_inv_d": simpson_inv_d,
            "detections": detections,
            "events": events
        })

    if not results:
        return pd.DataFrame(columns=empty_cols)

    df_metrics = pd.DataFrame(results)
    df_metrics = df_metrics.sort_values(["site", "period_start"]).reset_index(drop=True)

    return df_metrics

@st.cache_data
def prepare_long_term_summary(df_input, grain="M"):
    """
    Prépare un tableau résumé par période pour les dynamiques long terme.

    Pour chaque période, on calcule la moyenne inter-sites,
    l'écart-type, le nombre de sites et l'erreur standard.
    """
    # ---------------------------------------------------------
    # 1. Calcul détaillé par site et période
    # ---------------------------------------------------------
    df_long = prepare_long_term_indicators(df_input, grain=grain)

    if df_long is None or df_long.empty:
        return pd.DataFrame()

    # ---------------------------------------------------------
    # 2. Colonnes quantitatives à résumer
    # ---------------------------------------------------------
    metric_cols = [
        "richesse",
        "shannon",
        "pielou",
        "simpson_inv_d",
        "detections",
        "events"
    ]

    # ---------------------------------------------------------
    # 3. Résumé inter-sites par période
    # ---------------------------------------------------------
    summary_rows = []

    grouped = df_long.groupby(["period_label", "period_start"], dropna=False)

    for (period_label, period_start), group in grouped:
        row = {
            "period_label": period_label,
            "period_start": period_start
        }

        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()

            if len(values) == 0:
                mean_val = 0.0
                std_val = 0.0
                count_val = 0
                sem_val = 0.0
            else:
                mean_val = float(values.mean())
                std_val = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                count_val = int(len(values))
                sem_val = float(std_val / np.sqrt(count_val)) if count_val > 0 else 0.0

            row[f"{col}_mean"] = mean_val
            row[f"{col}_std"] = std_val
            row[f"{col}_count"] = count_val
            row[f"{col}_sem"] = sem_val

        summary_rows.append(row)

    if not summary_rows:
        return pd.DataFrame()

    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values("period_start").reset_index(drop=True)

    return df_summary

    # ---------------------------------------------------------
    # 1. Garde-fou : dataset vide ou absent
    # ---------------------------------------------------------
    if df_input is None or df_input.empty:
        return pd.DataFrame(columns=[
            "site",
            "period_label",
            "period_start",
            "richesse",
            "shannon",
            "pielou",
            "simpson_inv_d",
            "detections",
            "events"
        ])

    df_temp = df_input.copy()

    # ---------------------------------------------------------
    # 2. Vérification minimale des colonnes utiles
    # ---------------------------------------------------------
    required_cols = {"site", "startdate", "vernacular_name", "detection_count"}
    missing_cols = required_cols - set(df_temp.columns)

    if missing_cols:
        return pd.DataFrame(columns=[
            "site",
            "period_label",
            "period_start",
            "richesse",
            "shannon",
            "pielou",
            "simpson_inv_d",
            "detections",
            "events"
        ])

    # ---------------------------------------------------------
    # 3. Construction de la période selon le grain choisi
    # ---------------------------------------------------------
    # M = mois
    # Q = trimestre
    # Y = année
    # ---------------------------------------------------------
    if grain == "M":
        df_temp["period"] = df_temp["startdate"].dt.to_period("M")
    elif grain == "Q":
        df_temp["period"] = df_temp["startdate"].dt.to_period("Q")
    elif grain == "Y":
        df_temp["period"] = df_temp["startdate"].dt.to_period("Y")
    else:
        raise ValueError("grain doit valoir 'M', 'Q' ou 'Y'")

    # ---------------------------------------------------------
    # 4. Variables de période pour affichage et tri
    # ---------------------------------------------------------
    df_temp["period_start"] = df_temp["period"].dt.start_time
    df_temp["period_label"] = df_temp["period"].astype(str)

    # ---------------------------------------------------------
    # 5. Fonction locale de calcul des indicateurs
    # ---------------------------------------------------------
    def calc_metrics(group):
        """
        Calcule les métriques de biodiversité pour un site
        et une période donnés.
        """
        abundance = group.groupby("vernacular_name")["detection_count"].sum()
        total = abundance.sum()

        # Si aucune abondance exploitable
        if total <= 0 or len(abundance) == 0:
            return pd.Series({
                "richesse": 0,
                "shannon": 0.0,
                "pielou": 0.0,
                "simpson_inv_d": 0.0,
                "detections": 0.0,
                "events": int(len(group))
            })

        # Proportions relatives par espèce
        pi = abundance / total

        # Richesse spécifique = nombre d'espèces
        richesse = int(len(abundance))

        # Shannon
        shannon = float(-1 * (pi * np.log(pi + 1e-9)).sum())

        # Simpson inverse = nombre effectif d'espèces
        simpson_inv_d = float(1 / (pi ** 2).sum())

        # Piélou = Shannon / log(richesse)
        pielou = float(shannon / np.log(richesse) if richesse > 1 else 0.0)

        return pd.Series({
            "richesse": richesse,
            "shannon": shannon,
            "pielou": pielou,
            "simpson_inv_d": simpson_inv_d,
            "detections": float(total),
            "events": int(len(group))
        })

    # ---------------------------------------------------------
    # 6. Calcul des indicateurs par site et par période
    # ---------------------------------------------------------
    df_metrics = df_temp.groupby(
        ["site", "period_label", "period_start"]
    ).apply(
        calc_metrics,
        include_groups=False
    ).reset_index()

    # ---------------------------------------------------------
    # 7. Tri chronologique propre
    # ---------------------------------------------------------
    df_metrics = df_metrics.sort_values(
        ["site", "period_start"]
    ).reset_index(drop=True)

    return df_metrics

    def compute_shannon(group):
        """
        Calcule Shannon pour un groupe (site + semaine).
        """
        total = group['detection_count'].sum()

        if total == 0:
            return 0.0

        p_i = group['detection_count'] / total
        return float(-1 * (p_i * np.log(p_i + 1e-9)).sum())

    # Calcul final du Shannon hebdomadaire
    df_dist_shannon = (
        df_weekly_stats.groupby(['site', 'Semaine'])
        .apply(compute_shannon, include_groups=False)
        .reset_index(name='shannon_val')
    )

    return df_dist_shannon

" A SUPPRIMER"

def compute_indice_e1c(bootstrap_results, df_dist_shannon):
    """
    Calcule l’Indice E1C (Every1Counts), un score écologique global simplifié.

    Logique :
    - 40% Shannon : diversité globale
    - 30% Piélou  : équilibre de répartition
    - 30% stabilité temporelle : basée sur le coefficient de variation
      du Shannon hebdomadaire

    Le score final est ramené sur 100.

    Parameters
    ----------
    bootstrap_results : dict
        Résultats du bootstrap contenant notamment :
        - bootstrap_results['H'][0] = Shannon moyen
        - bootstrap_results['J'][0] = Piélou moyen

    df_dist_shannon : pd.DataFrame
        Distribution hebdomadaire du Shannon.

    Returns
    -------
    score_e1c : float
        Score global sur 100
    cv_shannon : float
        Coefficient de variation du Shannon hebdomadaire
    score_stabilite : float
        Sous-score de stabilité entre 0 et 1
    """
    if not bootstrap_results or df_dist_shannon.empty:
        return 0.0, np.nan, 0.0

    # ---------------------------------------------------------
    # 1. Composante diversité
    # ---------------------------------------------------------
    # On normalise Shannon sur 3, ce qui est cohérent
    # avec l’échelle de référence utilisée dans ton application.
    shannon_mean = bootstrap_results['H'][0]
    score_shannon = min(shannon_mean / 3.0, 1.0)

    # ---------------------------------------------------------
    # 2. Composante équilibre
    # ---------------------------------------------------------
    # Piélou est déjà compris entre 0 et 1.
    pielou_mean = bootstrap_results['J'][0]
    score_pielou = min(max(pielou_mean, 0), 1)

    # ---------------------------------------------------------
    # 3. Composante stabilité temporelle
    # ---------------------------------------------------------
    shannon_weekly = df_dist_shannon['shannon_val'].dropna()

    if len(shannon_weekly) >= 2 and shannon_weekly.mean() > 0:
        cv_shannon = shannon_weekly.std() / shannon_weekly.mean()

        # Plus le CV est faible, plus le site est stable
        score_stabilite = max(0.0, min(1.0, 1 - cv_shannon))
    else:
        cv_shannon = np.nan
        score_stabilite = 0.0

    # ---------------------------------------------------------
    # 4. Score final pondéré /100
    # ---------------------------------------------------------
    score_e1c = (
        score_shannon * 40 +
        score_pielou * 30 +
        score_stabilite * 30
    )

    return float(score_e1c), float(cv_shannon) if pd.notna(cv_shannon) else np.nan, float(score_stabilite)

# ---------------------------------------------------------
# FONCTION : calcul de l'indice E1C calibré SON
# ---------------------------------------------------------
def compute_indice_e1c_calibrated_sound(bootstrap_results, df_dist_shannon):
    """
    Calcule l'Indice E1C calibré pour les données acoustiques.

    Composantes :
    - Shannon  : 40%
    - Piélou   : 30%
    - Stabilité: 20%
    - Simpson  : 10%

    Chaque composante est ramenée sur 100,
    puis combinée selon les pondérations.
    """
    if not bootstrap_results or df_dist_shannon.empty:
        return {
            "score_e1c": 0.0,
            "cv_shannon": np.nan,
            "score_stabilite": 0.0,
            "score_shannon": 0.0,
            "score_pielou": 0.0,
            "score_simpson": 0.0
        }

    # ---------------------------------------------------------
    # 1. Valeurs bootstrapées
    # ---------------------------------------------------------
    shannon_mean = bootstrap_results['H'][0]
    pielou_mean = bootstrap_results['J'][0]
    simpson_mean = bootstrap_results['InvD'][0]

    # ---------------------------------------------------------
    # 2. Normalisation sur les références son
    # ---------------------------------------------------------
    score_shannon = normalize_score(
        shannon_mean,
        E1C_REFERENCE_SOUND["Shannon"]["min"],
        E1C_REFERENCE_SOUND["Shannon"]["max"]
    )

    score_pielou = normalize_score(
        pielou_mean,
        E1C_REFERENCE_SOUND["Pielou"]["min"],
        E1C_REFERENCE_SOUND["Pielou"]["max"]
    )

    score_simpson = normalize_score(
        simpson_mean,
        E1C_REFERENCE_SOUND["Simpson"]["min"],
        E1C_REFERENCE_SOUND["Simpson"]["max"]
    )

    # ---------------------------------------------------------
    # 3. Calcul de la stabilité temporelle
    # ---------------------------------------------------------
    shannon_weekly = df_dist_shannon['shannon_val'].dropna()

    if len(shannon_weekly) >= 2 and shannon_weekly.mean() > 0:
        cv_shannon = shannon_weekly.std() / shannon_weekly.mean()

        # CV faible = meilleure stabilité
        score_stabilite = max(0.0, min(100.0, (1 - min(cv_shannon, 1.0)) * 100))
    else:
        cv_shannon = np.nan
        score_stabilite = 0.0

    # ---------------------------------------------------------
    # 4. Score final pondéré
    # ---------------------------------------------------------
    score_e1c = (
        score_shannon * E1C_WEIGHTS_SOUND["Shannon"] +
        score_pielou * E1C_WEIGHTS_SOUND["Pielou"] +
        score_stabilite * E1C_WEIGHTS_SOUND["Stabilite"] +
        score_simpson * E1C_WEIGHTS_SOUND["Simpson"]
    )

    return {
        "score_e1c": float(score_e1c),
        "cv_shannon": float(cv_shannon) if pd.notna(cv_shannon) else np.nan,
        "score_stabilite": float(score_stabilite),
        "score_shannon": float(score_shannon),
        "score_pielou": float(score_pielou),
        "score_simpson": float(score_simpson)
    }

# ---------------------------------------------------------
# FONCTION : lecture du score E1C son
# ---------------------------------------------------------
def classify_e1c_sound(score_e1c):
    """
    Retourne une classe de lecture du score E1C son.
    """
    if score_e1c >= E1C_THRESHOLDS_SOUND["high"]:
        return "Excellent"
    elif score_e1c >= E1C_THRESHOLDS_SOUND["medium"]:
        return "Bon"
    elif score_e1c >= E1C_THRESHOLDS_SOUND["low"]:
        return "Intermédiaire"
    else:
        return "Faible"

def compute_species_dominance(df_input, top_n=3):
    """
    Calcule la part des espèces les plus dominantes dans le jeu de données.

    Logique :
    - on additionne les détections par espèce
    - on calcule la part représentée par les top espèces
    - plus cette part est élevée, plus le peuplement est dominé
      par un petit nombre d'espèces

    Parameters
    ----------
    df_input : pd.DataFrame
        Jeu de données filtré
    top_n : int
        Nombre d'espèces dominantes à considérer

    Returns
    -------
    dominance_ratio : float
        Part des top_n espèces dans l'abondance totale, entre 0 et 1
    df_top : pd.DataFrame
        Tableau des espèces dominantes
    """
    if df_input.empty:
        return 0.0, pd.DataFrame(columns=["vernacular_name", "detection_count"])

    df_counts = (
        df_input.groupby("vernacular_name")["detection_count"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    total = df_counts["detection_count"].sum()
    if total == 0:
        return 0.0, df_counts.head(top_n)

    top_sum = df_counts.head(top_n)["detection_count"].sum()
    dominance_ratio = top_sum / total

    return float(dominance_ratio), df_counts.head(top_n)

def compute_anthropic_pressure_index(df_input):
    """
    Calcule un indicateur simple de pression anthropique
    à partir de la proportion d'activité nocturne.

    Hypothèse :
    - une forte activité nocturne peut traduire un évitement
      de l'activité humaine diurne, surtout sur certains milieux

    Returns
    -------
    prop_nuit : float
        Pourcentage d'activité nocturne (0 à 100)
    pressure_score : float
        Score de pression normalisé entre 0 et 1
        - 0 = faible pression supposée
        - 1 = forte pression supposée
    """
    if df_input.empty or 'Heure' not in df_input.columns:
        return 0.0, 0.0

    df_temp = df_input.copy()

    n_nuit = len(df_temp[df_temp['Heure'].apply(lambda x: x < 7 or x >= 19)])
    n_total = len(df_temp)

    if n_total == 0:
        return 0.0, 0.0

    prop_nuit = (n_nuit / n_total) * 100

    # Transformation simple en score 0-1
    # 50% nocturne -> pression faible à modérée
    # 100% nocturne -> pression forte
    pressure_score = min(max((prop_nuit - 50) / 50, 0), 1)

    return float(prop_nuit), float(pressure_score)

def build_sites_map_figure(df_input, zoom=12):
    """
    Construit une carte des sites / hotspots à partir du dataframe filtré.

    La fonction :
    - vérifie la présence des colonnes nécessaires
    - force latitude / longitude / detection_count en numérique
    - agrège les détections par site
    - retourne une figure Plotly exploitable dans n'importe quel onglet

    Parameters
    ----------
    df_input : pd.DataFrame
        Jeu de données courant (filtré ou non)
    zoom : int or float
        Niveau de zoom initial de la carte

    Returns
    -------
    plotly.graph_objects.Figure or None
        Figure prête à afficher, ou None si impossible
    """
    import plotly.graph_objects as go

    required_cols = {'site', 'latitude', 'longitude'}
    if not required_cols.issubset(df_input.columns):
        return None

    df_map = df_input.copy()

    # Sécurisation des types
    df_map['latitude'] = pd.to_numeric(df_map['latitude'], errors='coerce')
    df_map['longitude'] = pd.to_numeric(df_map['longitude'], errors='coerce')

    # Si detection_count n'existe pas, chaque ligne vaut 1
    if 'detection_count' not in df_map.columns:
        df_map['detection_count'] = 1

    df_map['detection_count'] = pd.to_numeric(df_map['detection_count'], errors='coerce').fillna(1)

    # Agrégation
    df_map = (
        df_map[['site', 'latitude', 'longitude', 'detection_count']]
        .dropna(subset=['site', 'latitude', 'longitude'])
        .groupby(['site', 'latitude', 'longitude'], as_index=False)['detection_count']
        .sum()
    )

    if df_map.empty:
        return None

    det_min = df_map['detection_count'].min()
    det_max = df_map['detection_count'].max()

    if det_max == det_min:
        marker_sizes = np.full(len(df_map), 22)
    else:
        marker_sizes = 10 + (df_map['detection_count'] - det_min) / (det_max - det_min) * 28

    fig_map = go.Figure()

    fig_map.add_trace(go.Scattermapbox(
        lat=df_map['latitude'],
        lon=df_map['longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=marker_sizes,
            color=df_map['detection_count'],
            colorscale=[[0, C_ROSE], [1, C_ROUGE]],
            showscale=True,
            colorbar=dict(
                title="Détections",
                thickness=14,
                len=0.65,
                x=0.92
            ),
            opacity=0.9
        ),
        customdata=df_map[['site', 'detection_count']],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Détections : %{customdata[1]:.0f}<br>"
            "Lat : %{lat:.5f}<br>"
            "Lon : %{lon:.5f}"
            "<extra></extra>"
        ),
        showlegend=False
    ))

    fig_map.add_trace(go.Scattermapbox(
        lat=df_map['latitude'],
        lon=df_map['longitude'],
        mode='text',
        text=df_map['site'].astype(str),
        textposition='top right',
        textfont=dict(size=13, color='white', family='Arial Black'),
        hoverinfo='skip',
        showlegend=False
    ))

    fig_map.update_layout(
        mapbox=dict(
            style="white-bg",
            layers=[{
                "below": "traces",
                "sourcetype": "raster",
                "source": [
                    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ]
            }],
            center=dict(
                lat=float(df_map['latitude'].mean()),
                lon=float(df_map['longitude'].mean())
            ),
            zoom=zoom
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        height=420,
        paper_bgcolor=C_FOND,
        plot_bgcolor=C_FOND
    )

    return fig_map

@st.cache_data
def prepare_temporal_stability(df_input, grain="M"):
    """
    Calcule la stabilité temporelle de la communauté
    via la similarité de Jaccard entre périodes consécutives.

    Parameters
    ----------
    df_input : pd.DataFrame
        Jeu de données filtré.
    grain : str
        Grain temporel :
        - "M" = mois
        - "Q" = trimestre
        - "Y" = année

    Returns
    -------
    pd.DataFrame
        Tableau avec :
        - period_label
        - period_start
        - jaccard_prev
        - n_species
        - n_shared
    """
    # ---------------------------------------------------------
    # 1. Garde-fous
    # ---------------------------------------------------------
    if df_input is None or df_input.empty:
        return pd.DataFrame(columns=[
            "period_label",
            "period_start",
            "jaccard_prev",
            "n_species",
            "n_shared"
        ])

    df_temp = df_input.copy()

    required_cols = {"startdate", "vernacular_name"}
    if not required_cols.issubset(df_temp.columns):
        return pd.DataFrame(columns=[
            "period_label",
            "period_start",
            "jaccard_prev",
            "n_species",
            "n_shared"
        ])

    # ---------------------------------------------------------
    # 2. Sécurisation des dates
    # ---------------------------------------------------------
    df_temp["startdate"] = pd.to_datetime(df_temp["startdate"], errors="coerce")
    df_temp = df_temp.dropna(subset=["startdate", "vernacular_name"])

    if df_temp.empty:
        return pd.DataFrame(columns=[
            "period_label",
            "period_start",
            "jaccard_prev",
            "n_species",
            "n_shared"
        ])

    # ---------------------------------------------------------
    # 3. Construction des périodes
    # ---------------------------------------------------------
    if grain == "M":
        df_temp["period"] = df_temp["startdate"].dt.to_period("M")
    elif grain == "Q":
        df_temp["period"] = df_temp["startdate"].dt.to_period("Q")
    elif grain == "Y":
        df_temp["period"] = df_temp["startdate"].dt.to_period("Y")
    else:
        raise ValueError("grain doit valoir 'M', 'Q' ou 'Y'")

    df_temp["period_start"] = df_temp["period"].dt.start_time

    if grain == "Y":
        df_temp["period_label"] = df_temp["period"].dt.year.astype(str)
    elif grain == "Q":
        df_temp["period_label"] = df_temp["period"].apply(
            lambda p: f"T{p.quarter} {p.year}"
        )
    else:
        df_temp["period_label"] = df_temp["period"].astype(str)

    # ---------------------------------------------------------
    # 4. Liste des espèces par période
    # ---------------------------------------------------------
    period_species = (
        df_temp.groupby(["period_label", "period_start"])["vernacular_name"]
        .apply(lambda x: set(x.dropna().unique()))
        .reset_index(name="species_set")
        .sort_values("period_start")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 5. Similarité de Jaccard avec la période précédente
    # ---------------------------------------------------------
    rows = []

    prev_set = None

    for _, row in period_species.iterrows():
        current_set = row["species_set"]

        if prev_set is None:
            jaccard_prev = np.nan
            n_shared = np.nan
        else:
            intersection = len(current_set.intersection(prev_set))
            union = len(current_set.union(prev_set))
            jaccard_prev = intersection / union if union > 0 else np.nan
            n_shared = intersection

        rows.append({
            "period_label": row["period_label"],
            "period_start": row["period_start"],
            "jaccard_prev": jaccard_prev,
            "n_species": len(current_set),
            "n_shared": n_shared
        })

        prev_set = current_set

    return pd.DataFrame(rows)

@st.cache_data
def prepare_temporal_turnover(df_input, grain="M"):
    """
    Calcule le turnover écologique entre périodes consécutives.

    Pour chaque période, on compare la communauté observée
    à celle de la période précédente et on mesure :
    - le nombre d'espèces conservées
    - le nombre d'espèces gagnées
    - le nombre d'espèces perdues

    Parameters
    ----------
    df_input : pd.DataFrame
        Jeu de données filtré.
    grain : str
        Grain temporel :
        - "M" = mois
        - "Q" = trimestre
        - "Y" = année

    Returns
    -------
    pd.DataFrame
        Tableau avec :
        - period_label
        - period_start
        - status ("Conservées", "Gagnées", "Perdues")
        - n_species
        - gained
        - lost
        - retained
    """
    # ---------------------------------------------------------
    # 1. Garde-fous
    # ---------------------------------------------------------
    empty_cols = [
        "period_label",
        "period_start",
        "status",
        "n_species",
        "gained",
        "lost",
        "retained"
    ]

    if df_input is None or df_input.empty:
        return pd.DataFrame(columns=empty_cols)

    df_temp = df_input.copy()

    required_cols = {"startdate", "vernacular_name"}
    if not required_cols.issubset(df_temp.columns):
        return pd.DataFrame(columns=empty_cols)

    # ---------------------------------------------------------
    # 2. Sécurisation des dates
    # ---------------------------------------------------------
    df_temp["startdate"] = pd.to_datetime(df_temp["startdate"], errors="coerce")
    df_temp = df_temp.dropna(subset=["startdate", "vernacular_name"])

    if df_temp.empty:
        return pd.DataFrame(columns=empty_cols)

    # ---------------------------------------------------------
    # 3. Construction des périodes
    # ---------------------------------------------------------
    if grain == "M":
        df_temp["period"] = df_temp["startdate"].dt.to_period("M")
    elif grain == "Q":
        df_temp["period"] = df_temp["startdate"].dt.to_period("Q")
    elif grain == "Y":
        df_temp["period"] = df_temp["startdate"].dt.to_period("Y")
    else:
        raise ValueError("grain doit valoir 'M', 'Q' ou 'Y'")

    df_temp["period_start"] = df_temp["period"].dt.start_time

    if grain == "Y":
        df_temp["period_label"] = df_temp["period"].dt.year.astype(str)
    elif grain == "Q":
        df_temp["period_label"] = df_temp["period"].apply(
            lambda p: f"T{p.quarter} {p.year}"
        )
    else:
        df_temp["period_label"] = df_temp["period"].astype(str)

    # ---------------------------------------------------------
    # 4. Ensemble des espèces par période
    # ---------------------------------------------------------
    period_species = (
        df_temp.groupby(["period_label", "period_start"])["vernacular_name"]
        .apply(lambda x: set(x.dropna().unique()))
        .reset_index(name="species_set")
        .sort_values("period_start")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 5. Comparaison avec la période précédente
    # ---------------------------------------------------------
    rows = []
    prev_set = None

    for _, row in period_species.iterrows():
        current_set = row["species_set"]

        if prev_set is None:
            retained = np.nan
            gained = np.nan
            lost = np.nan
        else:
            retained = len(current_set.intersection(prev_set))
            gained = len(current_set - prev_set)
            lost = len(prev_set - current_set)

        rows.extend([
            {
                "period_label": row["period_label"],
                "period_start": row["period_start"],
                "status": "Conservées",
                "n_species": retained,
                "gained": gained,
                "lost": lost,
                "retained": retained
            },
            {
                "period_label": row["period_label"],
                "period_start": row["period_start"],
                "status": "Gagnées",
                "n_species": gained,
                "gained": gained,
                "lost": lost,
                "retained": retained
            },
            {
                "period_label": row["period_label"],
                "period_start": row["period_start"],
                "status": "Perdues",
                "n_species": lost,
                "gained": gained,
                "lost": lost,
                "retained": retained
            }
        ])

        prev_set = current_set

    df_turnover = pd.DataFrame(rows)

    return df_turnover

@st.cache_data
def prepare_temporal_dominance(df_input, grain="M", top_n=3):
    """
    Calcule la dynamique de domination / homogénéisation
    de la communauté dans le temps.

    Pour chaque période, on calcule :
    - la part des top_n espèces dominantes
    - le nombre d'espèces observées
    - l'équitabilité de Piélou
    - le nombre effectif d'espèces (Simpson inverse)

    Parameters
    ----------
    df_input : pd.DataFrame
        Jeu de données filtré.
    grain : str
        Grain temporel :
        - "M" = mois
        - "Q" = trimestre
        - "Y" = année
    top_n : int
        Nombre d'espèces dominantes à considérer.

    Returns
    -------
    pd.DataFrame
        Tableau avec une ligne par période.
    """
    # ---------------------------------------------------------
    # 1. Garde-fous
    # ---------------------------------------------------------
    empty_cols = [
        "period_label",
        "period_start",
        "dominance_ratio",
        "richesse",
        "pielou",
        "simpson_inv_d"
    ]

    if df_input is None or df_input.empty:
        return pd.DataFrame(columns=empty_cols)

    df_temp = df_input.copy()

    required_cols = {"startdate", "vernacular_name", "detection_count"}
    if not required_cols.issubset(df_temp.columns):
        return pd.DataFrame(columns=empty_cols)

    # ---------------------------------------------------------
    # 2. Sécurisation des types
    # ---------------------------------------------------------
    df_temp["startdate"] = pd.to_datetime(df_temp["startdate"], errors="coerce")
    df_temp["detection_count"] = pd.to_numeric(
        df_temp["detection_count"],
        errors="coerce"
    ).fillna(1)

    df_temp = df_temp.dropna(subset=["startdate", "vernacular_name"])

    if df_temp.empty:
        return pd.DataFrame(columns=empty_cols)

    # ---------------------------------------------------------
    # 3. Construction des périodes
    # ---------------------------------------------------------
    if grain == "M":
        df_temp["period"] = df_temp["startdate"].dt.to_period("M")
    elif grain == "Q":
        df_temp["period"] = df_temp["startdate"].dt.to_period("Q")
    elif grain == "Y":
        df_temp["period"] = df_temp["startdate"].dt.to_period("Y")
    else:
        raise ValueError("grain doit valoir 'M', 'Q' ou 'Y'")

    df_temp["period_start"] = df_temp["period"].dt.start_time

    if grain == "Y":
        df_temp["period_label"] = df_temp["period"].dt.year.astype(str)
    elif grain == "Q":
        df_temp["period_label"] = df_temp["period"].apply(
            lambda p: f"T{p.quarter} {p.year}"
        )
    else:
        df_temp["period_label"] = df_temp["period"].astype(str)

    # ---------------------------------------------------------
    # 4. Calcul des métriques par période
    # ---------------------------------------------------------
    rows = []

    grouped = df_temp.groupby(["period_label", "period_start"], dropna=False)

    for (period_label, period_start), group in grouped:
        abundance = (
            group.groupby("vernacular_name")["detection_count"]
            .sum()
            .sort_values(ascending=False)
        )

        total = abundance.sum()

        if total <= 0 or len(abundance) == 0:
            dominance_ratio = 0.0
            richesse = 0
            pielou = 0.0
            simpson_inv_d = 0.0
        else:
            top_sum = abundance.head(top_n).sum()
            dominance_ratio = float(top_sum / total)

            richesse = int(len(abundance))

            pi = abundance / total
            shannon = float(-1 * (pi * np.log(pi + 1e-9)).sum())
            pielou = float(shannon / np.log(richesse) if richesse > 1 else 0.0)
            simpson_inv_d = float(1 / (pi ** 2).sum())

        rows.append({
            "period_label": period_label,
            "period_start": period_start,
            "dominance_ratio": dominance_ratio,
            "richesse": richesse,
            "pielou": pielou,
            "simpson_inv_d": simpson_inv_d
        })

    df_dom = pd.DataFrame(rows)
    df_dom = df_dom.sort_values("period_start").reset_index(drop=True)

    return df_dom

@st.cache_data
def compute_ecosystem_trajectory(df_input, grain="M", dominance_top_n=3):
    """
    Produit une lecture synthétique de la trajectoire écologique
    à partir de plusieurs briques :
    - diversité long terme
    - stabilité temporelle
    - turnover
    - domination / homogénéisation

    Returns
    -------
    dict
        Dictionnaire avec :
        - diversity_trend
        - stability_level
        - turnover_balance
        - dominance_trend
        - trajectory_label
        - summary_text
    """
    # ---------------------------------------------------------
    # 1. Calcul des briques nécessaires
    # ---------------------------------------------------------
    df_long = prepare_long_term_summary(df_input, grain=grain)
    df_stability = prepare_temporal_stability(df_input, grain=grain)
    df_turnover = prepare_temporal_turnover(df_input, grain=grain)
    df_dom = prepare_temporal_dominance(df_input, grain=grain, top_n=dominance_top_n)

    # ---------------------------------------------------------
    # 2. Valeurs par défaut si données insuffisantes
    # ---------------------------------------------------------
    result = {
        "diversity_trend": "indéterminée",
        "stability_level": "indéterminée",
        "turnover_balance": "indéterminé",
        "dominance_trend": "indéterminée",
        "trajectory_label": "Lecture insuffisante",
        "summary_text": "Pas assez de recul pour produire une lecture synthétique robuste."
    }

    if df_long is None or df_long.empty:
        return result

    # ---------------------------------------------------------
    # 3. Tendance diversité (Shannon)
    # ---------------------------------------------------------
    if "shannon_mean" in df_long.columns and len(df_long) >= 2:
        first_shannon = df_long["shannon_mean"].iloc[0]
        last_shannon = df_long["shannon_mean"].iloc[-1]

        if first_shannon == 0 and last_shannon > 0:
            diversity_trend = "hausse"
        elif last_shannon > first_shannon * 1.05:
            diversity_trend = "hausse"
        elif last_shannon < first_shannon * 0.95:
            diversity_trend = "baisse"
        else:
            diversity_trend = "stable"
    else:
        diversity_trend = "indéterminée"

    # ---------------------------------------------------------
    # 4. Niveau de stabilité temporelle
    # ---------------------------------------------------------
    if df_stability is not None and not df_stability.empty:
        valid_jaccard = df_stability["jaccard_prev"].dropna()

        if len(valid_jaccard) > 0:
            mean_jaccard = valid_jaccard.mean()

            if mean_jaccard >= 0.70:
                stability_level = "élevée"
            elif mean_jaccard >= 0.50:
                stability_level = "intermédiaire"
            else:
                stability_level = "faible"
        else:
            stability_level = "indéterminée"
    else:
        stability_level = "indéterminée"

    # ---------------------------------------------------------
    # 5. Bilan turnover
    # ---------------------------------------------------------
    if df_turnover is not None and not df_turnover.empty:
        df_turnover_summary = (
            df_turnover.groupby(["period_label", "period_start"])
            .agg({
                "gained": "first",
                "lost": "first",
                "retained": "first"
            })
            .reset_index()
            .sort_values("period_start")
        )

        valid_turnover = df_turnover_summary.dropna(subset=["gained", "lost"])

        if len(valid_turnover) > 0:
            mean_gained = valid_turnover["gained"].mean()
            mean_lost = valid_turnover["lost"].mean()

            if mean_gained > mean_lost * 1.20:
                turnover_balance = "gains > pertes"
            elif mean_lost > mean_gained * 1.20:
                turnover_balance = "pertes > gains"
            else:
                turnover_balance = "équilibré"
        else:
            turnover_balance = "indéterminé"
    else:
        turnover_balance = "indéterminé"

    # ---------------------------------------------------------
    # 6. Tendance domination
    # ---------------------------------------------------------
    if df_dom is not None and not df_dom.empty and len(df_dom) >= 2:
        first_dom = df_dom["dominance_ratio"].iloc[0]
        last_dom = df_dom["dominance_ratio"].iloc[-1]

        if last_dom > first_dom * 1.10:
            dominance_trend = "hausse"
        elif last_dom < first_dom * 0.90:
            dominance_trend = "baisse"
        else:
            dominance_trend = "stable"
    else:
        dominance_trend = "indéterminée"

    # ---------------------------------------------------------
    # 7. Lecture synthétique
    # ---------------------------------------------------------
    if diversity_trend == "hausse" and dominance_trend == "baisse" and stability_level in ["élevée", "intermédiaire"]:
        trajectory_label = "Diversification progressive"
    elif diversity_trend == "stable" and dominance_trend == "stable" and stability_level == "élevée":
        trajectory_label = "Écosystème globalement stable"
    elif stability_level == "faible" and turnover_balance == "pertes > gains":
        trajectory_label = "Recomposition avec signal de vigilance"
    elif dominance_trend == "hausse" and diversity_trend in ["baisse", "stable"]:
        trajectory_label = "Simplification / homogénéisation probable"
    else:
        trajectory_label = "Trajectoire mixte ou intermédiaire"

    # ---------------------------------------------------------
    # 8. Texte client-friendly
    # ---------------------------------------------------------
    summary_parts = []

    if diversity_trend == "hausse":
        summary_parts.append("La diversité tend à progresser.")
    elif diversity_trend == "baisse":
        summary_parts.append("La diversité tend à reculer.")
    elif diversity_trend == "stable":
        summary_parts.append("La diversité reste globalement stable.")

    if stability_level == "élevée":
        summary_parts.append("La communauté apparaît stable dans le temps.")
    elif stability_level == "intermédiaire":
        summary_parts.append("La communauté présente une stabilité intermédiaire.")
    elif stability_level == "faible":
        summary_parts.append("La communauté est en recomposition marquée.")

    if turnover_balance == "gains > pertes":
        summary_parts.append("Les gains d'espèces dépassent globalement les pertes.")
    elif turnover_balance == "pertes > gains":
        summary_parts.append("Les pertes d'espèces dépassent globalement les gains.")
    elif turnover_balance == "équilibré":
        summary_parts.append("Les gains et pertes d'espèces restent globalement équilibrés.")

    if dominance_trend == "hausse":
        summary_parts.append("La domination de quelques espèces augmente.")
    elif dominance_trend == "baisse":
        summary_parts.append("La domination diminue, ce qui suggère une meilleure répartition.")
    elif dominance_trend == "stable":
        summary_parts.append("Le niveau de domination reste relativement stable.")

    summary_text = " ".join(summary_parts) if summary_parts else result["summary_text"]

    return {
        "diversity_trend": diversity_trend,
        "stability_level": stability_level,
        "turnover_balance": turnover_balance,
        "dominance_trend": dominance_trend,
        "trajectory_label": trajectory_label,
        "summary_text": summary_text
    }

@st.cache_data
def prepare_winners_losers_species(df_input, grain="M", min_total_detections=5):
    """
    Identifie les espèces en progression et en recul
    entre la première et la dernière période disponibles.

    Logique :
    - on agrège les détections par espèce et par période
    - on compare la première période à la dernière
    - on calcule l'écart absolu et l'évolution relative

    Parameters
    ----------
    df_input : pd.DataFrame
        Jeu de données filtré.
    grain : str
        Grain temporel :
        - "M" = mois
        - "Q" = trimestre
        - "Y" = année
    min_total_detections : int
        Seuil minimal de détections totales pour garder une espèce,
        afin d'éviter de commenter des signaux trop faibles.

    Returns
    -------
    pd.DataFrame
        Tableau avec :
        - vernacular_name
        - first_value
        - last_value
        - diff_abs
        - diff_rel
    """
    # ---------------------------------------------------------
    # 1. Garde-fous
    # ---------------------------------------------------------
    empty_cols = [
        "vernacular_name",
        "first_value",
        "last_value",
        "diff_abs",
        "diff_rel"
    ]

    if df_input is None or df_input.empty:
        return pd.DataFrame(columns=empty_cols)

    df_temp = df_input.copy()

    required_cols = {"startdate", "vernacular_name", "detection_count"}
    if not required_cols.issubset(df_temp.columns):
        return pd.DataFrame(columns=empty_cols)

    # ---------------------------------------------------------
    # 2. Sécurisation des types
    # ---------------------------------------------------------
    df_temp["startdate"] = pd.to_datetime(df_temp["startdate"], errors="coerce")
    df_temp["detection_count"] = pd.to_numeric(
        df_temp["detection_count"],
        errors="coerce"
    ).fillna(1)

    df_temp = df_temp.dropna(subset=["startdate", "vernacular_name"])

    if df_temp.empty:
        return pd.DataFrame(columns=empty_cols)

    # ---------------------------------------------------------
    # 3. Construction des périodes
    # ---------------------------------------------------------
    if grain == "M":
        df_temp["period"] = df_temp["startdate"].dt.to_period("M")
    elif grain == "Q":
        df_temp["period"] = df_temp["startdate"].dt.to_period("Q")
    elif grain == "Y":
        df_temp["period"] = df_temp["startdate"].dt.to_period("Y")
    else:
        raise ValueError("grain doit valoir 'M', 'Q' ou 'Y'")

    # ---------------------------------------------------------
    # 4. Agrégation par période et espèce
    # ---------------------------------------------------------
    df_species_period = (
        df_temp.groupby(["period", "vernacular_name"])["detection_count"]
        .sum()
        .reset_index()
    )

    if df_species_period.empty:
        return pd.DataFrame(columns=empty_cols)

    # ---------------------------------------------------------
    # 5. Pivot : lignes = espèces / colonnes = périodes
    # ---------------------------------------------------------
    pivot = df_species_period.pivot_table(
        index="vernacular_name",
        columns="period",
        values="detection_count",
        aggfunc="sum",
        fill_value=0
    )

    if pivot.shape[1] < 2:
        return pd.DataFrame(columns=empty_cols)

    # ---------------------------------------------------------
    # 6. Première et dernière période
    # ---------------------------------------------------------
    ordered_periods = sorted(pivot.columns)
    first_period = ordered_periods[0]
    last_period = ordered_periods[-1]

    # ---------------------------------------------------------
    # 7. Filtre sur les espèces suffisamment fréquentes
    # ---------------------------------------------------------
    total_counts = pivot.sum(axis=1)
    pivot = pivot[total_counts >= min_total_detections]

    if pivot.empty:
        return pd.DataFrame(columns=empty_cols)

    # ---------------------------------------------------------
    # 8. Calcul des variations
    # ---------------------------------------------------------
    df_delta = pd.DataFrame({
        "vernacular_name": pivot.index,
        "first_value": pivot[first_period].values,
        "last_value": pivot[last_period].values
    })

    df_delta["diff_abs"] = df_delta["last_value"] - df_delta["first_value"]

    # Variation relative simple
    # Si la valeur initiale = 0, on évite la division
    df_delta["diff_rel"] = np.where(
        df_delta["first_value"] > 0,
        (df_delta["last_value"] - df_delta["first_value"]) / df_delta["first_value"],
        np.nan
    )

    # Tri de confort
    df_delta = df_delta.sort_values("diff_abs", ascending=False).reset_index(drop=True)

    return df_delta

@st.cache_data
def prepare_temporal_bray_curtis(df_input, grain="M"):
    """
    Calcule la similarité de Bray-Curtis entre périodes consécutives.

    Retourne un score de similarité entre 0 et 1 :
    - 1 = structure très proche de la période précédente
    - 0 = structure très différente
    """
    empty_cols = ["period_label", "period_start", "bray_curtis_similarity"]

    if df_input is None or df_input.empty:
        return pd.DataFrame(columns=empty_cols)

    df_temp = df_input.copy()

    required_cols = {"startdate", "vernacular_name", "detection_count"}
    if not required_cols.issubset(df_temp.columns):
        return pd.DataFrame(columns=empty_cols)

    df_temp["startdate"] = pd.to_datetime(df_temp["startdate"], errors="coerce")
    df_temp["detection_count"] = pd.to_numeric(df_temp["detection_count"], errors="coerce").fillna(1)
    df_temp = df_temp.dropna(subset=["startdate", "vernacular_name"])

    if df_temp.empty:
        return pd.DataFrame(columns=empty_cols)

    if grain == "M":
        df_temp["period"] = df_temp["startdate"].dt.to_period("M")
    elif grain == "Q":
        df_temp["period"] = df_temp["startdate"].dt.to_period("Q")
    elif grain == "Y":
        df_temp["period"] = df_temp["startdate"].dt.to_period("Y")
    else:
        raise ValueError("grain doit valoir 'M', 'Q' ou 'Y'")

    df_temp["period_start"] = df_temp["period"].dt.start_time

    if grain == "Y":
        df_temp["period_label"] = df_temp["period"].dt.year.astype(str)
    elif grain == "Q":
        df_temp["period_label"] = df_temp["period"].apply(lambda p: f"T{p.quarter} {p.year}")
    else:
        df_temp["period_label"] = df_temp["period"].astype(str)

    pivot = df_temp.pivot_table(
        index="vernacular_name",
        columns=["period_label", "period_start"],
        values="detection_count",
        aggfunc="sum",
        fill_value=0
    )

    if pivot.shape[1] < 2:
        return pd.DataFrame(columns=empty_cols)

    ordered_cols = sorted(pivot.columns, key=lambda x: x[1])

    rows = []
    prev_vec = None
    prev_meta = None

    for col in ordered_cols:
        current_vec = pivot[col].astype(float)

        if prev_vec is None:
            similarity = np.nan
        else:
            numerator = np.abs(current_vec - prev_vec).sum()
            denominator = (current_vec + prev_vec).sum()
            dissimilarity = numerator / denominator if denominator > 0 else np.nan
            similarity = 1 - dissimilarity if pd.notna(dissimilarity) else np.nan

        rows.append({
            "period_label": col[0],
            "period_start": col[1],
            "bray_curtis_similarity": similarity
        })

        prev_vec = current_vec
        prev_meta = col

    return pd.DataFrame(rows)

# =========================================================
# PENTE DE TENDANCE LINÉAIRE
# =========================================================
@st.cache_data
def compute_linear_trend(df_input, x_col, y_col):
    """
    Calcule une tendance linéaire simple sur une série temporelle.

    Parameters
    ----------
    df_input : pd.DataFrame
        Tableau contenant la série.
    x_col : str
        Colonne temporelle (ordre des périodes).
    y_col : str
        Colonne quantitative à analyser.

    Returns
    -------
    dict
        {
            "slope": pente,
            "intercept": intercept,
            "r2": coefficient de détermination
        }
    """
    # ---------------------------------------------------------
    # 1. Garde-fous
    # ---------------------------------------------------------
    if df_input is None or df_input.empty:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan
        }

    if x_col not in df_input.columns or y_col not in df_input.columns:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan
        }

    # ---------------------------------------------------------
    # 2. Nettoyage des données
    # ---------------------------------------------------------
    df_temp = df_input[[x_col, y_col]].dropna().copy()

    if len(df_temp) < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan
        }

    df_temp = df_temp.sort_values(x_col).reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. Construction d'un temps numérique simple
    # ---------------------------------------------------------
    # On ne travaille pas ici avec des dates réelles, mais avec
    # l'ordre des périodes : 0, 1, 2, 3, ...
    # Cela suffit pour mesurer une tendance directionnelle.
    # ---------------------------------------------------------
    df_temp["t"] = np.arange(len(df_temp))

    x = df_temp["t"].values.astype(float)
    y = df_temp[y_col].values.astype(float)

    # ---------------------------------------------------------
    # 4. Régression linéaire simple
    # ---------------------------------------------------------
    slope, intercept = np.polyfit(x, y, 1)

    y_pred = slope * x + intercept

    # ---------------------------------------------------------
    # 5. Qualité d'ajustement (R²)
    # ---------------------------------------------------------
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot > 0:
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = np.nan

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2) if pd.notna(r2) else np.nan
    }

# ---------------------------------------------------------
# FONCTION : normaliser une valeur sur une échelle 0-100
# ---------------------------------------------------------
def normalize_score(value, vmin, vmax):
    """
    Ramène une valeur sur une échelle 0-100 entre une borne min et max.
    Toute valeur en dessous de min est ramenée à 0.
    Toute valeur au-dessus de max est ramenée à 100.
    """
    if pd.isna(value):
        return 0.0

    if vmax <= vmin:
        return 0.0

    score = (value - vmin) / (vmax - vmin)
    score = max(0.0, min(1.0, score))
    return float(score * 100)

# 8. CALCULS PRINCIPAUX
if not df.empty:
    with st.spinner('Calculs Bootstrap en cours...'):
        bootstrap_results = compute_bootstrap_stats(
            df,
            n_samples=BOOTSTRAP_CONFIG["n_samples"],
            n_iterations=BOOTSTRAP_CONFIG["n_iterations"]
        )
        mean_iajc, std_iajc = compute_bootstrap_iajc(
            df,
            n_iterations=BOOTSTRAP_CONFIG["iajc_iterations"]
        )

    counts_series = df.groupby('vernacular_name')['detection_count'].sum().sort_values(ascending=False)
    counts = counts_series.reset_index()
    counts.columns = ['Espèce', 'Nombre']

    top9 = counts.head(9)['Espèce'].tolist()
    color_map = {esp: PALETTE_ESPECES[i] for i, esp in enumerate(top9)}
    color_map["Autres"] = C_GRIS

        # ---------------------------------------------------------
    # Préparation des catégories graphiques
    # ---------------------------------------------------------
    # On conserve les 9 espèces les plus abondantes,
    # toutes les autres passent dans la catégorie "Autres"
    # pour garder des graphes lisibles.
    # ---------------------------------------------------------
    df['Espèce Graphique'] = df['vernacular_name'].apply(
        lambda x: x if x in top9 else "Autres"
    )

    # ---------------------------------------------------------
    # Standardisation de la colonne Semaine pour les graphiques
    # ---------------------------------------------------------
    # On réutilise la colonne week_start créée dans load_data()
    # afin d'éviter de recalculer la semaine avec une autre logique.
    # ---------------------------------------------------------
    df['Semaine'] = df['week_start']

    st.title("🎧 Tableaux de bord - Son")

    tab_global, tab_comparaison, tab_stats, tab_long_terme, tab_diagnostic, tab_export = st.tabs(
    ["📊 Dashboard Global", "🔬 Comparaison de Sites", "📈 Statistiques", "📆 Dynamiques long terme", "🧠 Diagnostic écologique", "📥 Export"]
    )

    # ---------------- TAB GLOBAL ----------------
    with tab_global:
        st.subheader("🛰️ Cartographie des Hotspots")

        if {'site', 'latitude', 'longitude', 'detection_count'}.issubset(df.columns):
            import plotly.graph_objects as go

            df_map = (
                df[['site', 'latitude', 'longitude', 'detection_count']]
                .dropna(subset=['site', 'latitude', 'longitude', 'detection_count'])
                .groupby(['site', 'latitude', 'longitude'], as_index=False)['detection_count']
                .sum()
            )

            if not df_map.empty:
                det_min = df_map['detection_count'].min()
                det_max = df_map['detection_count'].max()

                if det_max == det_min:
                    marker_sizes = np.full(len(df_map), 28)
                else:
                    marker_sizes = 12 + (df_map['detection_count'] - det_min) / (det_max - det_min) * 38

                fig_map = go.Figure()

                fig_map.add_trace(go.Scattermapbox(
                    lat=df_map['latitude'],
                    lon=df_map['longitude'],
                    mode='markers',
                    marker=go.scattermapbox.Marker(
                        size=marker_sizes,
                        color=df_map['detection_count'],
                        colorscale=[[0, C_ROSE], [1, C_ROUGE]],
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Détections", font=dict(color="white", size=20)),
                            tickfont=dict(color="white", size=20),
                            thickness=18,
                            len=0.75,
                            x=0.90,
                            y=0.5,
                            outlinewidth=0,
                            bgcolor="rgba(0,0,0,0)"
                        ),
                        opacity=0.9
                    ),
                    customdata=df_map[['site', 'detection_count']],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Détections : %{customdata[1]:.0f}<br>"
                        "Lat : %{lat:.5f}<br>"
                        "Lon : %{lon:.5f}"
                        "<extra></extra>"
                    ),
                    showlegend=False
                ))

                fig_map.add_trace(go.Scattermapbox(
                    lat=df_map['latitude'],
                    lon=df_map['longitude'],
                    mode='text',
                    text=df_map['site'].astype(str),
                    textposition='top right',
                    textfont=dict(size=14, color='white', family='Arial Black'),
                    hoverinfo='skip',
                    showlegend=False
                ))

                fig_map.update_layout(
                    mapbox=dict(
                        style="white-bg",
                        layers=[{
                            "below": "traces",
                            "sourcetype": "raster",
                            "source": [
                                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                            ]
                        }],
                        center=dict(lat=df_map['latitude'].mean(), lon=df_map['longitude'].mean()),
                        zoom=14.5
                    ),
                    margin=dict(r=0, t=0, l=0, b=0),
                    height=550,
                    paper_bgcolor=C_FOND,
                    plot_bgcolor=C_FOND
                )

                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning("Aucune donnée exploitable pour la cartographie.")
        else:
            st.error("❌ Colonnes nécessaires non trouvées : site, latitude, longitude, detection_count.")

        st.subheader("📌 Aperçu des données filtrées")
        date_debut = df['startdate'].min()
        date_fin = df['startdate'].max()
        n_jours_suivi = (date_fin.date() - date_debut.date()).days + 1

        str_debut = date_debut.strftime('%d/%m/%Y')
        str_fin = date_fin.strftime('%d/%m/%Y')
        total_ind = df['detection_count'].sum()
        total_ev = len(df)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("📅 Début", str_debut)
        k2.metric("🏁 Fin", str_fin)
        k3.metric("⏳ Suivi (jours)", n_jours_suivi)
        k4.metric("🐾 Observations", f"{total_ind:,}".replace(',', ' '))
        k5.metric("📸 Évènements", f"{total_ev:,}".replace(',', ' '))

        st.markdown("---")

        st.subheader("🔬 Synthèse des Indicateurs de Biodiversité")
        if bootstrap_results:
            res_s = f"{int(round(bootstrap_results['S'][0]))} ± {int(round(bootstrap_results['S'][1]))}"
            res_h = f"{bootstrap_results['H'][0]:.2f} ± {bootstrap_results['H'][1]:.2f}"
            res_invd = f"{bootstrap_results['InvD'][0]:.1f} ± {bootstrap_results['InvD'][1]:.1f}"
            res_j = f"{bootstrap_results['J'][0]:.2f} ± {bootstrap_results['J'][1]:.2f}"
            res_iajc = f"{mean_iajc:.1f} ± {std_iajc:.1f}"

            data_table = pd.DataFrame({
                "Indicateur": [
                    "Richesse Spécifique (S)",
                    "Indice de Shannon (H’)",
                    "Nombre effectif d’espèces (1 / D)",
                    "Équitabilité de Piélou (J)",
                    "Indice d'Activité (IAJC)"
                ],
                "Description": [
                    "Nombre d'espèces observées",
                    "Diversité richesse/abondance",
                    "Espèces dominantes",
                    "Equilibre répartition (0-1)",
                    "Activité normalisée jour/caméra"
                ],
                "Résultat (Moyenne ± σ)": [res_s, res_h, res_invd, res_j, res_iajc]
            })
            st.table(data_table)

            st.info(
                "Méthode bootstrap : pour garantir des indicateurs comparables entre sites, "
                f"nous simulons {BOOTSTRAP_CONFIG['n_iterations']} échantillons aléatoires de "
                f"{BOOTSTRAP_CONFIG['n_samples']} observations. "
                "Chaque indicateur est recalculé à chaque itération, puis moyenné. "
                "Cette approche permet de lisser les effets du hasard et des variations "
                "d’effort d’échantillonnage.")

        st.markdown("---")

        col1, col2 = st.columns([1.2, 1.4])

        with col1:
            st.subheader("🍰 Répartition de l'abondance")

            df_pie = df.groupby('Espèce Graphique')['detection_count'].sum().reset_index()

            fig_pie = px.pie(
                df_pie,
                values='detection_count',
                names='Espèce Graphique',
                hole=0.6,
                color='Espèce Graphique',
                color_discrete_map=color_map,
                template="none"
            )

            fig_pie.update_layout(
                paper_bgcolor=C_FOND,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.4,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(t=20, b=20, l=20, r=20)
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("📊 Abondance par Site")
            df_sites = df.groupby(['site', 'Espèce Graphique'])['detection_count'].sum().reset_index()
            fig_bar = px.bar(
                df_sites,
                x="site",
                y="detection_count",
                color="Espèce Graphique",
                barmode="stack",
                template="none",
                color_discrete_map=color_map
            )
            fig_bar.update_layout(
                xaxis={'categoryorder': 'total descending', 'tickangle': -45},
                paper_bgcolor=C_FOND,
                plot_bgcolor=C_FOND,
                margin=dict(b=100)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("⏰ Activité (Cumul individus)")
            df_24h = df.groupby(['Heure', 'Espèce Graphique'])['detection_count'].sum().reset_index()
            fig_24h = px.bar(
                df_24h,
                x="Heure",
                y="detection_count",
                color="Espèce Graphique",
                barmode="stack",
                template="none",
                color_discrete_map=color_map
            )
            fig_24h.update_layout(
                paper_bgcolor=C_FOND,
                plot_bgcolor=C_FOND,
                xaxis=dict(tickmode='linear', dtick=2)
            )
            st.plotly_chart(fig_24h, use_container_width=True)

        with col4:
            st.subheader("📈 Évolution temporelle")
            df_temp = df.groupby(['Semaine', 'Espèce Graphique'])['detection_count'].sum().reset_index()
            fig_time = px.bar(
                df_temp,
                x="Semaine",
                y="detection_count",
                color="Espèce Graphique",
                barmode="stack",
                template="none",
                color_discrete_map=color_map
            )
            fig_time.update_layout(
                paper_bgcolor=C_FOND,
                plot_bgcolor=C_FOND,
                xaxis=dict(
                    tickformat="%b %Y",
                    dtick="M1",
                    tickangle=-45
                )
            )
            st.plotly_chart(fig_time, use_container_width=True)

        st.markdown("---")

        st.subheader(f"📋 Inventaire des observations ({len(df['vernacular_name'].unique())} espèces)")

        counts_inventaire = df.groupby('vernacular_name')['detection_count'].sum().sort_values(
            ascending=False
        ).reset_index()
        counts_inventaire.columns = ['Espèce', 'Nombre']

        if espece_sidebar != "Toutes les espèces":
            c_disp = counts_inventaire[counts_inventaire['Espèce'] == espece_sidebar].copy()
        else:
            c_disp = counts_inventaire.copy()

        st.dataframe(
            c_disp,
            use_container_width=True,
            hide_index=True,
            height=380,
            column_config={
                "Espèce": st.column_config.TextColumn(
                    "Espèce",
                    help="Nom vernaculaire de l'espèce",
                    width="large",
                ),
                "Nombre": st.column_config.NumberColumn(
                    "Nombre d'individus",
                    format="%d",
                    width="medium",
                )
            }
        )

    # ---------------- TAB COMPARAISON ----------------
    with tab_comparaison:
        st.subheader("🔬 Positionnement du site sur les indices de biodiversité")

        st.info(LEGENDE_SITES)

        df_comp = df_base_date.copy()
        if filtre_sauvage == "Sauvages uniquement":
            df_comp = df_comp[~df_comp['vernacular_name'].isin(domestiques)]

        if df_comp.empty:
            st.warning("⚠️ Aucune donnée disponible pour la période sélectionnée.")
        else:
            with st.spinner("Calcul des indices bootstrap..."):
                stats_comp = compute_bootstrap_stats(
                    df_comp,
                    n_samples=BOOTSTRAP_CONFIG["n_samples"],
                    n_iterations=BOOTSTRAP_CONFIG["n_iterations"]
                )
                iajc_m, iajc_s = compute_bootstrap_iajc(
                    df_comp,
                    n_iterations=BOOTSTRAP_CONFIG["iajc_iterations"]
                )

            if stats_comp:
                indices_a_afficher = [
                    ("Shannon", stats_comp["H"][0]),
                    ("Richesse", round(stats_comp["S"][0])),
                    ("InvD", stats_comp["InvD"][0]),
                    ("Pielou", stats_comp["J"][0]),
                    ("IAJC", iajc_m),  # 👈 AJOUT
                ]

                for cle_indice, valeur_site in indices_a_afficher:
                    ref = REFERENCES_INDICES[cle_indice]

                    fig_indice = comp.generer_graphe_indice(
                        val_site=valeur_site,
                        nom_indice=ref["nom"],
                        sites_fixes=ref["sites"],
                        mode=ref["mode"],
                        min_tick_force=ref["min_tick"],
                        max_tick_force=ref["max_tick"]
                    )

                    st.pyplot(fig_indice, clear_figure=True)

        st.subheader("📊 Comparaison Inter-Sites : Diversité vs Activité")

        if df_base_date.empty:
            st.warning("⚠️ Aucune donnée disponible pour la période sélectionnée. Vérifiez vos filtres.")
        else:
            with st.spinner('Calcul des points de comparaison SITE...'):
                stats_b_toutes = compute_bootstrap_stats(
                    df_base_date,
                    n_samples=BOOTSTRAP_CONFIG["n_samples"],
                    n_iterations=BOOTSTRAP_CONFIG["n_iterations"]
                )
                iajc_b_toutes_m, iajc_b_toutes_s = compute_bootstrap_iajc(
                    df_base_date,
                    n_iterations=BOOTSTRAP_CONFIG["iajc_iterations"]
                )

            if stats_b_toutes:
                # Références lues depuis la configuration du début de programme
                ref_shannon = {s["label"]: s for s in REFERENCES_INDICES["Shannon"]["sites"]}
                ref_iajc = {s["label"]: s for s in REFERENCES_INDICES["IAJC"]["sites"]}

                comparison_data = [
                    {
                        "Site": "SITE",
                        "Shannon": stats_b_toutes['H'][0],
                        "S_err": stats_b_toutes['H'][1],
                        "IAJC": iajc_b_toutes_m,
                        "I_err": iajc_b_toutes_s
                    },

                    {
                        "Site": "Etrechy",
                        "Shannon": ref_shannon["ET"]["score"],
                        "S_err": ref_shannon["ET"]["err"],
                        "IAJC": ref_iajc["ET"]["score"],
                        "I_err": ref_iajc["ET"]["err"]
                    },
                    {
                        "Site": "Lavallière",
                        "Shannon": ref_shannon["LV"]["score"],
                        "S_err": ref_shannon["LV"]["err"],
                        "IAJC": ref_iajc["LV"]["score"],
                        "I_err": ref_iajc["LV"]["err"]
                    },
                    {
                        "Site": "La Peyruche",
                        "Shannon": ref_shannon["LP"]["score"],
                        "S_err": ref_shannon["LP"]["err"],
                        "IAJC": ref_iajc["LP"]["score"],
                        "I_err": ref_iajc["LP"]["err"]
                    }
                ]

                df_plot = pd.DataFrame(comparison_data)

                charte_map = {
                    "SITE": C_ROUGE,
                    "Etrechy": C_VERT_SOMBRE,
                    "Lavallière": C_BLEU,
                    "La Peyruche": C_JAUNE
                }

                fig_corr = px.scatter(
                    df_plot,
                    x="IAJC",
                    y="Shannon",
                    text="Site",
                    color="Site",
                    color_discrete_map=charte_map,
                    error_x="I_err",
                    error_y="S_err",
                    template="none"
                )

                fig_corr.update_traces(
                    textposition='top center',
                    marker=dict(size=16, line=dict(width=1.5, color='white')),
                    unselected=dict(marker=dict(opacity=0.3))
                )

                fig_corr.update_layout(
                    paper_bgcolor=C_FOND,
                    plot_bgcolor=C_FOND,
                    xaxis_title="Indice d'Activité (IAJC) →",
                    yaxis_title="Indice de Shannon (H') →",
                    height=700,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.15,
                        xanchor="center",
                        x=0.5,
                        title=None
                    ),
                    margin=dict(b=150, t=50)
                )

                st.plotly_chart(fig_corr, use_container_width=True)
                st.info("💡 Ce graphique utilise les couleurs officielles de votre charte. Les points 'SITE' s'actualisent selon les dates sélectionnées.")
            else:
                st.warning("⚠️ Pas assez de données pour construire la comparaison inter-sites.")

        # ---------------- TAB STATS ----------------
    with tab_stats:
        st.subheader("📦 Distribution Hebdomadaire de la Diversité (Shannon)")

        # ---------------------------------------------------------
        # Calcul de la distribution hebdomadaire de Shannon
        # ---------------------------------------------------------
        # On utilise la fonction mutualisée pour :
        # - éviter de dupliquer le code
        # - garantir la cohérence avec le reste du dashboard
        # - améliorer les performances (cache possible)
        # ---------------------------------------------------------
        df_dist_shannon = compute_weekly_shannon_distribution(df)

        site_groups = [df_dist_shannon[df_dist_shannon['site'] == s]['shannon_val'] for s in df_dist_shannon['site'].unique()]

        anova_result = "Non calculable"
        p_value = 1.0

        if len(site_groups) > 1:
            f_stat, p_value = scipy_stats.f_oneway(*site_groups)
            significatif = "SIGNIFICATIVE" if p_value < 0.05 else "NON SIGNIFICATIVE"
            anova_result = f"ANOVA : F={f_stat:.2f}, p-value={p_value:.4e} ({significatif})"

        site_means = df_dist_shannon.groupby('site')['shannon_val'].mean().sort_values(ascending=False)
        order_sites = site_means.index.tolist()

        fig_box = px.box(
            df_dist_shannon,
            x='site',
            y='shannon_val',
            color='site',
            category_orders={'site': order_sites},
            color_discrete_sequence=PALETTE_ESPECES,
            template="none",
            points="all"
        )

        fig_box.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.1,
            text=f"<b>{anova_result}</b>",
            showarrow=False,
            font=dict(size=14, color=C_VERT_SOMBRE if p_value < 0.05 else C_ROUGE),
            bgcolor="white",
            opacity=0.8
        )

        fig_box.update_layout(
            paper_bgcolor=C_FOND,
            plot_bgcolor=C_FOND,
            xaxis_title="Sites (classés par moyenne décroissante)",
            yaxis_title="Indice de Shannon (H')",
            showlegend=False,
            xaxis={'tickangle': -45},
            height=650,
            margin=dict(t=80)
        )

        st.plotly_chart(fig_box, use_container_width=True)

        if p_value < 0.05:
            st.success("✅ **Résultat significatif** : Il existe une différence réelle de biodiversité entre les sites (p < 0.05).")
        else:
            st.info("⚖️ **Résultat non significatif** : Les variations observées peuvent être dues au hasard (p > 0.05).")

        st.markdown("---")
        st.subheader("📊 Comparaison des Moyennes (Tukey HSD)")
        fig_tukey = plot_tukey_shannon(df)
        st.plotly_chart(fig_tukey, use_container_width=True, key="chart_tukey")

        st.markdown("---")
        st.subheader("📊 Distribution des Rangs (Kruskal-Wallis)")
        p_val_kw, kw_letters, fig_kw = plot_kruskal_shannon(df)
        st.plotly_chart(fig_kw, use_container_width=True, key="chart_kruskal")

        st.markdown("---")
        st.subheader("🕒 Rythmes d'Activité")

        liste_especes_stats = ["Toutes les espèces"] + sorted(
            df['vernacular_name']
            .dropna()  # supprime les None / NaN
            .astype(str)  # sécurise en string
            .unique()
            .tolist()
        )
        espece_selectionnee_stats = st.selectbox("🎯 Filtrer par espèce pour l'activité :", liste_especes_stats)

        if espece_selectionnee_stats == "Toutes les espèces":
            df_heatmap = df
        else:
            df_heatmap = df[df['vernacular_name'] == espece_selectionnee_stats]

        if not df_heatmap.empty:
            st.write(f"Analyse de l'activité pour : **{espece_selectionnee_stats}**")
            fig_activity = plot_activity_heatmap(df_heatmap)
            st.plotly_chart(fig_activity, use_container_width=True)
        else:
            st.warning("Aucune donnée pour cette espèce.")

        st.markdown("---")
        st.subheader("🌳 Classification Hiérarchique")

        if df['site'].nunique() > 2:
            fig_dendro = plot_dendrogramme_jaccard(df)
            st.plotly_chart(fig_dendro, use_container_width=True)

            st.info("""
**Interprétation :** Plus les sites sont regroupés tôt (branches basses), 
plus leur composition en espèces est identique. C'est le test visuel 
ultime pour la similitude inter-sites.
""")
        else:
            st.warning("⚠️ Le dendrogramme nécessite au moins 3 sites pour être pertinent.")

        st.markdown("---")
        st.subheader("✨ Espèces Indicatrices (Analyse IndVal)")
        st.write("""
Cette analyse identifie l'espèce la plus **caractéristique** de chaque site. 
Un score élevé (proche de 1) signifie que l'espèce est à la fois unique à ce site et observée très régulièrement.
""")

        fig_indval = plot_indicator_species(df)
        st.plotly_chart(fig_indval, use_container_width=True, key="chart_indval")

        st.info("""
**Interprétation :** Si un site a un score élevé pour une espèce sensible (ex: le Renard), 
cela renforce la valeur écologique de ce hotspot précis par rapport aux autres.
""")

        st.markdown("---")
        st.subheader("🌓 Rythme Circadien : Jour vs Nuit")
        st.write("""
Ce graphique compare la proportion de détections diurnes (07h-19h) et nocturnes. 
Un site avec une **forte dominance nocturne** (VIOLET) peut indiquer un évitement de l'activité humaine diurne.
""")

        fig_dn = plot_diurne_nocturne(df)
        st.plotly_chart(fig_dn, use_container_width=True, key="chart_diurne_nocturne")

        nuit_max = df[df['Heure'].apply(lambda x: x < 7 or x >= 19)]
        prop_nuit = (len(nuit_max) / len(df)) * 100 if len(df) > 0 else 0

        st.info(f"""
**Diagnostic flash :** Sur l'ensemble des données sélectionnées, **{prop_nuit:.1f}%** de l'activité est nocturne. 
Si ce chiffre dépasse 70% sur un site forestier, la quiétude diurne est probablement altérée.
""")

        st.markdown("---")
        st.subheader("⏳ Complétude et Saturation de l'Inventaire")

        fig_acc, score_completude = plot_courbe_accumulation(df)
        st.plotly_chart(fig_acc, use_container_width=True)

        if score_completude > 90:
            st.success(f"🎯 **Inventaire robuste ({score_completude:.1f}%)** : La courbe a atteint un plateau. Il est peu probable de découvrir de nouvelles espèces sans changer de méthode de capture.")
        elif score_completude > 70:
            st.info(f"📊 **Inventaire satisfaisant ({score_completude:.1f}%)** : L'essentiel de la biodiversité est capté, mais quelques espèces rares pourraient encore apparaître.")
        else:
            st.warning(f"⚠️ **Inventaire incomplet ({score_completude:.1f}%)** : La courbe continue de monter. Une prolongation du suivi est recommandée pour stabiliser les indices.")

    # ---------------- TAB DYNAMIQUES LONG TERME ----------------
    with tab_long_terme:
        st.subheader("📆 Dynamiques de biodiversité à long terme")

        st.write(
            "Cet onglet permet de suivre l'évolution des grands indicateurs "
            "à un pas de temps plus large que la semaine : mois, trimestre ou année."
        )

        # ---------------------------------------------------------
        # Choix du grain temporel
        # ---------------------------------------------------------
        grain_label = st.selectbox(
            "Choisir le grain temporel :",
            ["Mois", "Trimestre", "Année"],
            index=0,
            key="long_term_grain_son"
        )

        grain_map = {
            "Mois": "M",
            "Trimestre": "Q",
            "Année": "Y"
        }

        grain_code = grain_map[grain_label]

        # ---------------------------------------------------------
        # Résumé statistique par période
        # ---------------------------------------------------------
        df_long_summary = prepare_long_term_summary(df, grain=grain_code)

        if df_long_summary is None or df_long_summary.empty:
            st.warning("⚠️ Aucune donnée disponible pour construire les dynamiques long terme.")
        else:
            # ---------------------------------------------------------
            # Choix de l'indicateur
            # ---------------------------------------------------------
            metric_choice = st.selectbox(
                "Choisir l'indicateur à visualiser :",
                [
                    "Richesse spécifique",
                    "Shannon",
                    "Piélou",
                    "Nombre effectif (1 / D)",
                    "Détections",
                    "Évènements"
                ],
                index=1,
                key="long_term_metric_son"
            )

            metric_config = {
                "Richesse spécifique": {
                    "mean": "richesse_mean",
                    "sem": "richesse_sem",
                    "title": "📊 Évolution de la richesse spécifique",
                    "ylabel": "Nombre d'espèces",
                    "color": C_VERT_SOMBRE
                },
                "Shannon": {
                    "mean": "shannon_mean",
                    "sem": "shannon_sem",
                    "title": "📊 Évolution de l'indice de Shannon",
                    "ylabel": "Indice de Shannon (H')",
                    "color": C_BLEU
                },
                "Piélou": {
                    "mean": "pielou_mean",
                    "sem": "pielou_sem",
                    "title": "📊 Évolution de l'équitabilité de Piélou",
                    "ylabel": "Indice de Piélou (J)",
                    "color": C_VIOLET
                },
                "Nombre effectif (1 / D)": {
                    "mean": "simpson_inv_d_mean",
                    "sem": "simpson_inv_d_sem",
                    "title": "📊 Évolution du nombre effectif d'espèces",
                    "ylabel": "Simpson inverse (1 / D)",
                    "color": C_JAUNE
                },
                "Détections": {
                    "mean": "detections_mean",
                    "sem": "detections_sem",
                    "title": "📊 Évolution du nombre de détections",
                    "ylabel": "Nombre de détections",
                    "color": C_ROUGE
                },
                "Évènements": {
                    "mean": "events_mean",
                    "sem": "events_sem",
                    "title": "📊 Évolution du nombre d'évènements",
                    "ylabel": "Nombre d'évènements",
                    "color": C_ROSE
                }
            }

            conf = metric_config[metric_choice]

                        # ---------------------------------------------------------
            # Histogramme principal avec correction axe X
            # ---------------------------------------------------------

            # 🔧 IMPORTANT : forcer les labels en texte
            df_long_summary["period_label"] = df_long_summary["period_label"].astype(str)

            fig_long = px.bar(
                df_long_summary,
                x="period_label",
                y=conf["mean"],
                template="none",
                text_auto=".2f"
            )

            fig_long.update_traces(
                marker_color=conf["color"],
                marker_line_color=C_VERT_SOMBRE,
                marker_line_width=1.2,
                error_y=dict(
                    type="data",
                    array=df_long_summary[conf["sem"]],
                    visible=True,
                    color=C_GRIS,
                    thickness=1.5,
                    width=5
                ),
                textposition="outside"
            )

            y_max = (
                df_long_summary[conf["mean"]] + df_long_summary[conf["sem"]]
            ).max()

            fig_long.update_layout(
                title=conf["title"],
                xaxis_title=grain_label,
                yaxis_title=conf["ylabel"],
                paper_bgcolor=C_FOND,
                plot_bgcolor=C_FOND,
                height=550,
                showlegend=False,
                xaxis=dict(tickangle=-45),
                yaxis=dict(range=[0, y_max * 1.20 if y_max > 0 else 1])
            )

            # 🔥 CORRECTION PRINCIPALE : axe catégoriel
            fig_long.update_xaxes(type="category")

            st.plotly_chart(fig_long, use_container_width=True)

            st.markdown("---")

            # ---------------------------------------------------------
            # Vue multi-indicateurs en histogrammes
            # ---------------------------------------------------------
            st.markdown("### 🔬 Vue synthétique multi-indicateurs")

            c1, c2 = st.columns(2)

            with c1:
                fig_shannon = px.bar(
                    df_long_summary,
                    x="period_label",
                    y="shannon_mean",
                    template="none",
                    text_auto=".2f"
                )
                fig_shannon.update_traces(
                    marker_color=C_ROSE,
                    marker_line_color=C_VERT_SOMBRE,
                    marker_line_width=1.2,
                    error_y=dict(
                        type="data",
                        array=df_long_summary["shannon_sem"],
                        visible=True,
                        color=C_GRIS
                    ),
                    textposition="outside"
                )
                fig_shannon.update_layout(
                    title="Indice de Shannon",
                    xaxis_title=grain_label,
                    yaxis_title="H'",
                    paper_bgcolor=C_FOND,
                    plot_bgcolor=C_FOND,
                    height=400,
                    showlegend=False,
                    xaxis=dict(tickangle=-45)
                )
                st.plotly_chart(fig_shannon, use_container_width=True)

            with c2:
                fig_richesse = px.bar(
                    df_long_summary,
                    x="period_label",
                    y="richesse_mean",
                    template="none",
                    text_auto=".0f"
                )
                fig_richesse.update_traces(
                    marker_color=C_VERT_SOMBRE,
                    marker_line_color=C_VERT_SOMBRE,
                    marker_line_width=1.2,
                    error_y=dict(
                        type="data",
                        array=df_long_summary["richesse_sem"],
                        visible=True,
                        color=C_GRIS
                    ),
                    textposition="outside"
                )
                fig_richesse.update_layout(
                    title="Richesse spécifique",
                    xaxis_title=grain_label,
                    yaxis_title="Nombre d'espèces",
                    paper_bgcolor=C_FOND,
                    plot_bgcolor=C_FOND,
                    height=400,
                    showlegend=False,
                    xaxis=dict(tickangle=-45)
                )
                st.plotly_chart(fig_richesse, use_container_width=True)

            c3, c4 = st.columns(2)

            with c3:
                fig_pielou = px.bar(
                    df_long_summary,
                    x="period_label",
                    y="pielou_mean",
                    template="none",
                    text_auto=".2f"
                )
                fig_pielou.update_traces(
                    marker_color=C_VIOLET,
                    marker_line_color=C_VERT_SOMBRE,
                    marker_line_width=1.2,
                    error_y=dict(
                        type="data",
                        array=df_long_summary["pielou_sem"],
                        visible=True,
                        color=C_GRIS
                    ),
                    textposition="outside"
                )
                fig_pielou.update_layout(
                    title="Équitabilité de Piélou",
                    xaxis_title=grain_label,
                    yaxis_title="J",
                    paper_bgcolor=C_FOND,
                    plot_bgcolor=C_FOND,
                    height=400,
                    showlegend=False,
                    xaxis=dict(tickangle=-45)
                )
                st.plotly_chart(fig_pielou, use_container_width=True)

            with c4:
                fig_simpson = px.bar(
                    df_long_summary,
                    x="period_label",
                    y="simpson_inv_d_mean",
                    template="none",
                    text_auto=".2f"
                )
                fig_simpson.update_traces(
                    marker_color=C_JAUNE,
                    marker_line_color=C_VERT_SOMBRE,
                    marker_line_width=1.2,
                    error_y=dict(
                        type="data",
                        array=df_long_summary["simpson_inv_d_sem"],
                        visible=True,
                        color=C_GRIS
                    ),
                    textposition="outside"
                )
                fig_simpson.update_layout(
                    title="Nombre effectif d'espèces (1 / D)",
                    xaxis_title=grain_label,
                    yaxis_title="1 / D",
                    paper_bgcolor=C_FOND,
                    plot_bgcolor=C_FOND,
                    height=400,
                    showlegend=False,
                    xaxis=dict(tickangle=-45)
                )
                st.plotly_chart(fig_simpson, use_container_width=True)

            st.markdown("---")

            # ---------------------------------------------------------
            # Lecture automatique simple
            # ---------------------------------------------------------
            st.markdown("### 🧠 Lecture automatique")

            comments = []

            for metric_name, mean_col in [
                ("la richesse spécifique", "richesse_mean"),
                ("l'indice de Shannon", "shannon_mean"),
                ("l'équitabilité de Piélou", "pielou_mean"),
                ("le nombre effectif d'espèces", "simpson_inv_d_mean")
            ]:
                df_trend = df_long_summary[["period_start", mean_col]].dropna().copy()

                if len(df_trend) >= 2:
                    first_val = df_trend[mean_col].iloc[0]
                    last_val = df_trend[mean_col].iloc[-1]

                    if first_val == 0 and last_val > 0:
                        comments.append(f"Tendance à l'amélioration pour {metric_name}.")
                    elif last_val > first_val * 1.05:
                        comments.append(f"Tendance à l'amélioration pour {metric_name}.")
                    elif last_val < first_val * 0.95:
                        comments.append(f"Tendance à la baisse pour {metric_name}.")
                    else:
                        comments.append(f"Stabilité globale pour {metric_name}.")
                else:
                    comments.append(f"Pas assez de recul pour interpréter {metric_name}.")

            st.write(" ".join(comments))

            st.markdown("---")
            st.markdown("### 🌿 Stabilité de la communauté dans le temps")

            st.write(
                "Ce graphique mesure à quel point la communauté observée "
                "ressemble à celle de la période précédente. "
                "Plus le score est élevé, plus la composition en espèces est stable."
            )

            # ---------------------------------------------------------
            # Calcul de la stabilité temporelle
            # ---------------------------------------------------------
            df_stability = prepare_temporal_stability(df, grain=grain_code)

            if df_stability is None or df_stability.empty:
                st.warning("⚠️ Pas assez de données pour calculer la stabilité temporelle.")
            else:
                df_stability["period_label_str"] = df_stability["period_label"].astype(str)

                # ---------------------------------------------------------
                # Histogramme de similarité temporelle
                # ---------------------------------------------------------
                fig_stability = px.bar(
                    df_stability,
                    x="period_label_str",
                    y="jaccard_prev",
                    template="none",
                    text_auto=".2f"
                )

                fig_stability.update_traces(
                    marker_color=C_BLEU,
                    marker_line_color=C_VERT_SOMBRE,
                    marker_line_width=1.2,
                    textposition="outside"
                )

                # Courbe de tendance superposée
                fig_stability.add_scatter(
                    x=df_stability["period_label_str"],
                    y=df_stability["jaccard_prev"],
                    mode="lines+markers",
                    name="Tendance",
                    line=dict(color=C_ROUGE, width=3),
                    marker=dict(color=C_ROUGE, size=8)
                )

                fig_stability.update_layout(
                    title="Similarité avec la période précédente (indice de Jaccard)",
                    xaxis_title=grain_label,
                    yaxis_title="Similarité (0 à 1)",
                    paper_bgcolor=C_FOND,
                    plot_bgcolor=C_FOND,
                    height=500,
                    yaxis=dict(range=[0, 1.05]),
                    xaxis=dict(tickangle=-45),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    )
                )

                fig_stability.update_xaxes(type="category")

                st.plotly_chart(fig_stability, use_container_width=True)

                # ---------------------------------------------------------
                # Lecture automatique simple
                # ---------------------------------------------------------
                st.markdown("#### 🧠 Lecture écologique")

                valid_vals = df_stability["jaccard_prev"].dropna()

                if len(valid_vals) == 0:
                    st.info("Pas assez de recul pour interpréter la stabilité temporelle.")
                else:
                    mean_jaccard = valid_vals.mean()
                    min_jaccard = valid_vals.min()

                    comments = []

                    if mean_jaccard >= 0.7:
                        comments.append("La communauté apparaît globalement stable dans le temps.")
                    elif mean_jaccard >= 0.5:
                        comments.append("La communauté montre une stabilité intermédiaire, avec quelques phases de recomposition.")
                    else:
                        comments.append("La communauté est peu stable dans le temps, ce qui suggère une forte recomposition.")

                    if min_jaccard < 0.35:
                        comments.append("Au moins une rupture marquée de composition est visible entre deux périodes consécutives.")

                    st.write(" ".join(comments))

            st.markdown("---")
            st.markdown("### 🔄 Turnover écologique (espèces gagnées / perdues)")

            st.write(
                "Ce graphique montre comment la communauté évolue d'une période à l'autre : "
                "quelles espèces sont conservées, quelles nouvelles espèces apparaissent, "
                "et quelles espèces disparaissent."
            )


            # ---------------------------------------------------------
            # Calcul du turnover temporel
            # ---------------------------------------------------------
            df_turnover = prepare_temporal_turnover(df, grain=grain_code)

            if df_turnover is None or df_turnover.empty:
                st.warning("⚠️ Pas assez de données pour calculer le turnover écologique.")
            else:
                # ---------------------------------------------------------
                # Sécurisation axe X
                # ---------------------------------------------------------
                df_turnover["period_label_str"] = df_turnover["period_label"].astype(str)

                # ---------------------------------------------------------
                # Histogramme empilé
                # ---------------------------------------------------------
                color_map_turnover = {
                    "Conservées": C_VERT_SOMBRE,
                    "Gagnées": C_BLEU,
                    "Perdues": C_ROUGE
                }

                fig_turnover = px.bar(
                    df_turnover,
                    x="period_label_str",
                    y="n_species",
                    color="status",
                    barmode="stack",
                    template="none",
                    color_discrete_map=color_map_turnover
                )

                fig_turnover.update_layout(
                    title="Renouvellement de la communauté entre périodes consécutives",
                    xaxis_title=grain_label,
                    yaxis_title="Nombre d'espèces",
                    paper_bgcolor=C_FOND,
                    plot_bgcolor=C_FOND,
                    height=520,
                    legend_title="Statut",
                    xaxis=dict(tickangle=-45)
                )

                fig_turnover.update_xaxes(type="category")

                st.plotly_chart(fig_turnover, use_container_width=True)

                # ---------------------------------------------------------
                # Lecture automatique simple
                # ---------------------------------------------------------
                st.markdown("#### 🧠 Lecture écologique du turnover")

                # On travaille sur une table à une ligne par période
                df_turnover_summary = (
                    df_turnover.groupby(["period_label", "period_start"])
                    .agg({
                        "gained": "first",
                        "lost": "first",
                        "retained": "first"
                    })
                    .reset_index()
                    .sort_values("period_start")
                )

                valid_turnover = df_turnover_summary.dropna(subset=["gained", "lost", "retained"]).copy()

                if valid_turnover.empty:
                    st.info("Pas assez de recul pour interpréter le turnover écologique.")
                else:
                    mean_gained = valid_turnover["gained"].mean()
                    mean_lost = valid_turnover["lost"].mean()
                    max_lost = valid_turnover["lost"].max()

                    comments = []

                    if mean_gained > mean_lost * 1.2:
                        comments.append("La communauté gagne globalement plus d'espèces qu'elle n'en perd.")
                    elif mean_lost > mean_gained * 1.2:
                        comments.append("La communauté perd globalement plus d'espèces qu'elle n'en gagne.")
                    else:
                        comments.append("Les gains et pertes d'espèces restent globalement équilibrés.")

                    if max_lost >= 3:
                        comments.append("Au moins une période montre une perte marquée d'espèces, à surveiller.")

                    st.write(" ".join(comments))

            st.markdown("---")
            st.markdown("### 🌊 Stabilité des abondances (Bray-Curtis temporel)")

            st.write(
                "Ce graphique mesure à quel point la structure d'abondance de la communauté "
                "ressemble à celle de la période précédente."
            )

            df_bray = prepare_temporal_bray_curtis(df, grain=grain_code)

            if df_bray is None or df_bray.empty:
                st.warning("⚠️ Pas assez de données pour calculer Bray-Curtis.")
            else:
                df_bray["period_label_str"] = df_bray["period_label"].astype(str)

                fig_bray = px.bar(
                    df_bray,
                    x="period_label_str",
                    y="bray_curtis_similarity",
                    template="none",
                    text_auto=".2f"
                )

                fig_bray.update_traces(
                    marker_color=C_MAUVE,
                    marker_line_color=C_VERT_SOMBRE,
                    marker_line_width=1.2,
                    textposition="outside"
                )

                fig_bray.add_scatter(
                    x=df_bray["period_label_str"],
                    y=df_bray["bray_curtis_similarity"],
                    mode="lines+markers",
                    name="Tendance",
                    line=dict(color=C_ROUGE, width=3),
                    marker=dict(color=C_ROUGE, size=8)
                )

                fig_bray.update_layout(
                    title="Similarité d'abondance avec la période précédente (Bray-Curtis)",
                    xaxis_title=grain_label,
                    yaxis_title="Similarité (0 à 1)",
                    paper_bgcolor=C_FOND,
                    plot_bgcolor=C_FOND,
                    height=500,
                    yaxis=dict(range=[0, 1.05]),
                    xaxis=dict(tickangle=-45)
                )
                fig_bray.update_xaxes(type="category")

                st.plotly_chart(fig_bray, use_container_width=True)

                                # ---------------------------------------------------------
                # Lecture automatique Bray-Curtis
                # ---------------------------------------------------------
                st.markdown("#### 🧠 Lecture écologique (Bray-Curtis)")

                valid_vals = df_bray["bray_curtis_similarity"].dropna()

                if len(valid_vals) == 0:
                    st.info("Pas assez de données pour interpréter la stabilité des abondances.")
                else:
                    mean_bc = valid_vals.mean()
                    min_bc = valid_vals.min()

                    comments = []

                    # -----------------------------------------------------
                    # Niveau global de stabilité des abondances
                    # -----------------------------------------------------
                    if mean_bc >= 0.75:
                        comments.append("La structure d'abondance des espèces reste globalement très stable dans le temps.")
                    elif mean_bc >= 0.55:
                        comments.append("La structure d'abondance présente des variations modérées entre périodes.")
                    else:
                        comments.append("La structure d'abondance est instable, ce qui suggère une recomposition importante.")

                    # -----------------------------------------------------
                    # Détection d'événement ponctuel
                    # -----------------------------------------------------
                    if min_bc < 0.40:
                        worst_period = df_bray.loc[df_bray["bray_curtis_similarity"].idxmin(), "period_label"]
                        comments.append(
                            f"Un changement marqué des abondances est observé autour de {worst_period}."
                        )

                    # -----------------------------------------------------
                    # Lecture croisée avec Jaccard (si dispo)
                    # -----------------------------------------------------
                    if "jaccard_prev" in df_stability.columns:
                        valid_j = df_stability["jaccard_prev"].dropna()

                        if len(valid_j) > 0:
                            mean_j = valid_j.mean()

                            if mean_j >= 0.7 and mean_bc < 0.6:
                                comments.append(
                                    "Les espèces présentes restent globalement les mêmes, mais leur abondance relative évolue."
                                )

                    st.write(" ".join(comments))

#DOMINATION HOMOGENISATION

            st.markdown("---")
            st.markdown("### 🧬 Domination / homogénéisation de la communauté")

            st.write(
                "Ce bloc permet de suivre si la communauté devient de plus en plus "
                "dominée par un petit nombre d'espèces, ou au contraire plus équilibrée."
            )

            # ---------------------------------------------------------
            # Paramètre utilisateur : nombre d'espèces dominantes
            # ---------------------------------------------------------
            top_n_dom = st.slider(
                "Nombre d'espèces dominantes à considérer :",
                min_value=3,
                max_value=20,
                value=10,
                step=1,
                key="dominance_top_n"
            )

            # ---------------------------------------------------------
            # Calcul des métriques de domination
            # ---------------------------------------------------------
            df_dom = prepare_temporal_dominance(
                df,
                grain=grain_code,
                top_n=top_n_dom
            )

            if df_dom is None or df_dom.empty:
                st.warning("⚠️ Pas assez de données pour calculer la domination écologique.")
            else:
                # ---------------------------------------------------------
                # Sécurisation axe X
                # ---------------------------------------------------------
                df_dom["period_label_str"] = df_dom["period_label"].astype(str)

                # ---------------------------------------------------------
                # Graphique principal : part des espèces dominantes
                # ---------------------------------------------------------
                fig_dom = px.bar(
                    df_dom,
                    x="period_label_str",
                    y="dominance_ratio",
                    template="none",
                    text_auto=".2f"
                )

                fig_dom.update_traces(
                    marker_color=C_VIOLET,
                    marker_line_color=C_VERT_SOMBRE,
                    marker_line_width=1.2,
                    textposition="outside"
                )

                # Courbe de tendance
                fig_dom.add_scatter(
                    x=df_dom["period_label_str"],
                    y=df_dom["dominance_ratio"],
                    mode="lines+markers",
                    name="Tendance",
                    line=dict(color=C_ROUGE, width=3),
                    marker=dict(color=C_ROUGE, size=8)
                )

                fig_dom.update_layout(
                    title=f"Part des {top_n_dom} espèces dominantes",
                    xaxis_title=grain_label,
                    yaxis_title="Part de l'abondance totale",
                    paper_bgcolor=C_FOND,
                    plot_bgcolor=C_FOND,
                    height=500,
                    yaxis=dict(range=[0, 1.05]),
                    xaxis=dict(tickangle=-45),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    )
                )

                fig_dom.update_xaxes(type="category")

                st.plotly_chart(fig_dom, use_container_width=True)

                # ---------------------------------------------------------
                # Vue croisée domination / équilibre
                # ---------------------------------------------------------
                st.markdown("#### 🔬 Lecture croisée structure / équilibre")

                c1, c2 = st.columns(2)

                with c1:
                    fig_pielou_dom = px.bar(
                        df_dom,
                        x="period_label_str",
                        y="pielou",
                        template="none",
                        text_auto=".2f"
                    )
                    fig_pielou_dom.update_traces(
                        marker_color=C_BLEU,
                        marker_line_color=C_VERT_SOMBRE,
                        marker_line_width=1.2,
                        textposition="outside"
                    )
                    fig_pielou_dom.add_scatter(
                        x=df_dom["period_label_str"],
                        y=df_dom["pielou"],
                        mode="lines+markers",
                        name="Tendance",
                        line=dict(color=C_ROUGE, width=3),
                        marker=dict(color=C_ROUGE, size=8)
                    )
                    fig_pielou_dom.update_layout(
                        title="Équitabilité de Piélou",
                        xaxis_title=grain_label,
                        yaxis_title="J",
                        paper_bgcolor=C_FOND,
                        plot_bgcolor=C_FOND,
                        height=400,
                        yaxis=dict(range=[0, 1.05]),
                        showlegend=False,
                        xaxis=dict(tickangle=-45)
                    )
                    fig_pielou_dom.update_xaxes(type="category")
                    st.plotly_chart(fig_pielou_dom, use_container_width=True)

                with c2:
                    fig_simpson_dom = px.bar(
                        df_dom,
                        x="period_label_str",
                        y="simpson_inv_d",
                        template="none",
                        text_auto=".2f"
                    )
                    fig_simpson_dom.update_traces(
                        marker_color=C_JAUNE,
                        marker_line_color=C_VERT_SOMBRE,
                        marker_line_width=1.2,
                        textposition="outside"
                    )
                    fig_simpson_dom.add_scatter(
                        x=df_dom["period_label_str"],
                        y=df_dom["simpson_inv_d"],
                        mode="lines+markers",
                        name="Tendance",
                        line=dict(color=C_ROUGE, width=3),
                        marker=dict(color=C_ROUGE, size=8)
                    )
                    fig_simpson_dom.update_layout(
                        title="Nombre effectif d'espèces (1 / D)",
                        xaxis_title=grain_label,
                        yaxis_title="1 / D",
                        paper_bgcolor=C_FOND,
                        plot_bgcolor=C_FOND,
                        height=400,
                        showlegend=False,
                        xaxis=dict(tickangle=-45)
                    )
                    fig_simpson_dom.update_xaxes(type="category")
                    st.plotly_chart(fig_simpson_dom, use_container_width=True)

                # ---------------------------------------------------------
                # Lecture automatique simple
                # ---------------------------------------------------------
                st.markdown("#### 🧠 Lecture écologique de la domination")

                comments = []

                if len(df_dom) >= 2:
                    first_dom = df_dom["dominance_ratio"].iloc[0]
                    last_dom = df_dom["dominance_ratio"].iloc[-1]

                    first_pielou = df_dom["pielou"].iloc[0]
                    last_pielou = df_dom["pielou"].iloc[-1]

                    if last_dom > first_dom * 1.10:
                        comments.append("La communauté devient progressivement plus dominée par un petit nombre d'espèces.")
                    elif last_dom < first_dom * 0.90:
                        comments.append("La domination diminue, ce qui suggère une communauté plus équilibrée.")
                    else:
                        comments.append("Le niveau global de domination reste relativement stable.")

                    if last_pielou > first_pielou * 1.05:
                        comments.append("L'équitabilité progresse, signe d'une meilleure répartition des abondances.")
                    elif last_pielou < first_pielou * 0.95:
                        comments.append("L'équitabilité recule, ce qui peut traduire une homogénéisation croissante.")

                if comments:
                    st.write(" ".join(comments))
                else:
                    st.info("Pas assez de recul pour interpréter la domination écologique.")

            st.markdown("---")
            st.markdown("### 🧭 Trajectoire écologique synthétique")

            st.write(
                "Cette synthèse combine plusieurs dimensions de l'écosystème : "
                "diversité, stabilité de la communauté, turnover et domination."
            )

            # ---------------------------------------------------------
            # Calcul de la trajectoire synthétique
            # ---------------------------------------------------------
            trajectory = compute_ecosystem_trajectory(
                df,
                grain=grain_code,
                dominance_top_n=top_n_dom
            )

            # ---------------------------------------------------------
            # Résumé visuel sous forme de métriques
            # ---------------------------------------------------------
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Diversité", trajectory["diversity_trend"].capitalize())

            with c2:
                st.metric("Stabilité", trajectory["stability_level"].capitalize())

            with c3:
                st.metric("Turnover", trajectory["turnover_balance"].capitalize())

            with c4:
                st.metric("Domination", trajectory["dominance_trend"].capitalize())

            st.markdown("#### ✅ Lecture synthétique")
            st.success(trajectory["trajectory_label"])

            st.markdown("#### 🧠 Interprétation")
            st.write(trajectory["summary_text"])

            st.markdown("---")
            st.markdown("### 🐾 Espèces en progression / en recul")

            st.write(
                "Ce bloc compare la première et la dernière période disponibles "
                "pour identifier les espèces qui progressent le plus et celles qui reculent."
            )

            # ---------------------------------------------------------
            # Paramètre utilisateur : seuil minimal de détections
            # ---------------------------------------------------------
            # Permet d'éviter de commenter des espèces trop rares
            # (signal statistique peu robuste)
            # ---------------------------------------------------------
            min_det_species = st.slider(
                "Nombre minimum d’observations pour analyser une espèce",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="winners_losers_min_det"
            )

            # ---------------------------------------------------------
            # Calcul des gagnantes / perdantes
            # ---------------------------------------------------------
            df_delta_species = prepare_winners_losers_species(
                df,
                grain=grain_code,
                min_total_detections=min_det_species
            )

            if df_delta_species is None or df_delta_species.empty:
                st.warning("⚠️ Pas assez de données pour identifier les espèces en progression ou en recul.")
            else:
                # ---------------------------------------------------------
                # Sélection TOP 5 gagnantes et perdantes
                # ---------------------------------------------------------
                df_winners = (
                    df_delta_species
                    .sort_values("diff_abs", ascending=False)
                    .head(5)
                    .sort_values("diff_abs", ascending=True)
                )

                df_losers = (
                    df_delta_species
                    .sort_values("diff_abs", ascending=True)
                    .head(5)
                )

                c1, c2 = st.columns(2)

                # =========================================================
                # 📈 GRAPHE ESPÈCES EN PROGRESSION
                # =========================================================
                with c1:
                    st.markdown("#### 📈 Espèces en progression")

                    fig_winners = px.bar(
                        df_winners,
                        x="diff_abs",
                        y="vernacular_name",
                        orientation="h",
                        template="none",
                        text="diff_abs"
                    )

                    fig_winners.update_traces(
                        marker_color=C_BLEU,
                        marker_line_color=C_VERT_SOMBRE,
                        marker_line_width=1.2,
                        textposition="outside"
                    )

                    # 🔥 Correction principale d'affichage :
                    # - suppression "Espèce"
                    # - augmentation marge gauche
                    # - meilleure lisibilité des noms
                    fig_winners.update_layout(
                        title="Gains entre la première et la dernière période",
                        xaxis_title="Variation absolue des détections",
                        yaxis_title="",
                        paper_bgcolor=C_FOND,
                        plot_bgcolor=C_FOND,
                        height=460,
                        showlegend=False,
                        margin=dict(l=180, r=40, t=70, b=60)
                    )

                    fig_winners.update_yaxes(
                        automargin=True
                    )

                    st.plotly_chart(fig_winners, use_container_width=True)

                # =========================================================
                # 📉 GRAPHE ESPÈCES EN RECUL
                # =========================================================
                with c2:
                    st.markdown("#### 📉 Espèces en recul")

                    fig_losers = px.bar(
                        df_losers,
                        x="diff_abs",
                        y="vernacular_name",
                        orientation="h",
                        template="none",
                        text="diff_abs"
                    )

                    fig_losers.update_traces(
                        marker_color=C_ROUGE,
                        marker_line_color=C_VERT_SOMBRE,
                        marker_line_width=1.2,
                        textposition="outside"
                    )

                    # 🔥 Même correction ici
                    fig_losers.update_layout(
                        title="Pertes entre la première et la dernière période",
                        xaxis_title="Variation absolue des détections",
                        yaxis_title="",
                        paper_bgcolor=C_FOND,
                        plot_bgcolor=C_FOND,
                        height=460,
                        showlegend=False,
                        margin=dict(l=180, r=40, t=70, b=60)
                    )

                    fig_losers.update_yaxes(
                        automargin=True
                    )

                    st.plotly_chart(fig_losers, use_container_width=True)

                # ---------------------------------------------------------
                # Lecture automatique simple (message client)
                # ---------------------------------------------------------
                st.markdown("#### 🧠 Lecture écologique simple")

                winner_names = df_winners[df_winners["diff_abs"] > 0]["vernacular_name"].tolist()
                loser_names = df_losers[df_losers["diff_abs"] < 0]["vernacular_name"].tolist()

                comments = []

                if winner_names:
                    comments.append(
                        "Espèces en progression : " + ", ".join(winner_names[:3]) + "."
                    )

                if loser_names:
                    comments.append(
                        "Espèces en recul : " + ", ".join(loser_names[:3]) + "."
                    )

                if comments:
                    st.write(" ".join(comments))
                else:
                    st.info("Aucune variation marquée n'est détectée sur les espèces les plus fréquentes.")

        # ---------------- TAB DIAGNOSTIC ----------------
    with tab_diagnostic:
        st.subheader("🧠 Diagnostic écologique")

        # ---------------------------------------------------------
        # 1. Calcul du Shannon hebdomadaire (stabilité)
        # ---------------------------------------------------------
        df_dist_shannon_diag = compute_weekly_shannon_distribution(df)

        # ---------------------------------------------------------
        # Calcul de l’indice E1C calibré SON
        # ---------------------------------------------------------
        e1c_results = compute_indice_e1c_calibrated_sound(
            bootstrap_results,
            df_dist_shannon_diag
        )

        score_e1c = e1c_results["score_e1c"]
        cv_shannon = e1c_results["cv_shannon"]
        score_stabilite = e1c_results["score_stabilite"]
        score_shannon = e1c_results["score_shannon"]
        score_pielou = e1c_results["score_pielou"]
        score_simpson = e1c_results["score_simpson"]

        classe_e1c = classify_e1c_sound(score_e1c)

        # ---------------------------------------------------------
        # 3. Affichage des indicateurs principaux
        # ---------------------------------------------------------
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Indice E1C",
                f"{score_e1c:.0f} / 100"
            )

        with c2:
            st.metric(
                "Classe écologique",
                classe_e1c
            )

        with c3:
            if pd.notna(cv_shannon):
                st.metric(
                    "Stabilité temporelle (CV Shannon)",
                    f"{cv_shannon:.2f}"
                )
            else:
                st.metric(
                    "Stabilité temporelle (CV Shannon)",
                    "n.c."
                )

        with c4:
            st.metric(
                "Score de stabilité",
                f"{score_stabilite:.0f} / 100"
            )

        # ---------------------------------------------------------
        # 4. Interprétation globale
        # ---------------------------------------------------------
        if score_e1c >= E1C_THRESHOLDS_SOUND["high"]:
            st.success("🟢 Indice E1C excellent : site parmi les meilleurs niveaux observés dans le référentiel acoustique.")
        elif score_e1c >= E1C_THRESHOLDS_SOUND["medium"]:
            st.success("🟢 Indice E1C bon : site de bon niveau écologique dans le référentiel acoustique.")
        elif score_e1c >= E1C_THRESHOLDS_SOUND["low"]:
            st.warning("🟠 Indice E1C intermédiaire : site fonctionnel mais encore perfectible.")
        else:
            st.error("🔴 Indice E1C faible : site acoustiquement simplifié ou sous pression.")

        # ---------------------------------------------------------
        # 5. Décomposition du score E1C
        # ---------------------------------------------------------
        st.markdown("### 🧩 Décomposition de l’Indice E1C")

        df_e1c_components = pd.DataFrame({
            "Composante": [
                "Diversité (Shannon)",
                "Équilibre (Piélou)",
                "Structure (Simpson 1/D)",
                "Stabilité temporelle"
            ],
            "Score /100": [
                score_shannon,
                score_pielou,
                score_simpson,
                score_stabilite
            ]
        })

        fig_e1c = px.bar(
            df_e1c_components,
            x="Composante",
            y="Score /100",
            text="Score /100",
            template="none"
        )

        fig_e1c.update_traces(
            texttemplate="%{y:.0f}",
            textposition="outside"
        )

        fig_e1c.update_layout(
            paper_bgcolor=C_FOND,
            plot_bgcolor=C_FOND,
            yaxis=dict(range=[0, 110]),
            showlegend=False
        )

        st.plotly_chart(fig_e1c, use_container_width=True)

        # ---------------------------------------------------------
        # 5. Dominance des espèces
        # ---------------------------------------------------------
        dominance_ratio, df_top_dom = compute_species_dominance(df, top_n=3)

        st.markdown("### 🐾 Dominance des espèces")

        c4, c5 = st.columns([1, 1.5])

        with c4:
            st.metric(
                "Part des 3 espèces dominantes",
                f"{dominance_ratio * 100:.1f}%"
            )

            if dominance_ratio < 0.40:
                st.success("🟢 Peuplement bien réparti")
            elif dominance_ratio < 0.60:
                st.warning("🟠 Dominance modérée")
            else:
                st.error("🔴 Forte dominance de quelques espèces")

        with c5:
            st.dataframe(
                df_top_dom.rename(columns={
                    "vernacular_name": "Espèce",
                    "detection_count": "Détections"
                }),
                use_container_width=True,
                hide_index=True
            )

        # ---------------------------------------------------------
        # 6. Pression anthropique / quiétude
        # ---------------------------------------------------------
        prop_nuit_diag, pressure_score = compute_anthropic_pressure_index(df)

        st.markdown("### 🌙 Pression anthropique / quiétude")

        c6, c7 = st.columns(2)

        with c6:
            st.metric(
                "Activité nocturne",
                f"{prop_nuit_diag:.1f}%"
            )

            if prop_nuit_diag < 50:
                st.success("🟢 Activité majoritairement diurne")
            elif prop_nuit_diag < 70:
                st.warning("🟠 Nocturnité modérée")
            else:
                st.error("🔴 Forte nocturnité")

        with c7:
            st.metric(
                "Score de pression",
                f"{pressure_score:.2f}"
            )

            if pressure_score < 0.25:
                st.success("🟢 Pression anthropique faible supposée")
            elif pressure_score < 0.50:
                st.warning("🟠 Pression anthropique modérée")
            else:
                st.error("🔴 Pression anthropique élevée supposée")

        # ---------------------------------------------------------
        # 7. Lecture écologique automatique
        # ---------------------------------------------------------
        commentaires = []

        if bootstrap_results:
            if bootstrap_results['H'][0] >= 1.8:
                commentaires.append("La diversité spécifique est élevée.")
            elif bootstrap_results['H'][0] >= 1.3:
                commentaires.append("La diversité spécifique est intermédiaire.")
            else:
                commentaires.append("La diversité spécifique reste limitée.")

            if bootstrap_results['J'][0] >= 0.6:
                commentaires.append("La répartition des espèces est équilibrée.")
            elif bootstrap_results['J'][0] >= 0.4:
                commentaires.append("La répartition est modérée.")
            else:
                commentaires.append("Quelques espèces dominent fortement le site.")

        if pd.notna(cv_shannon):
            if cv_shannon < 0.2:
                commentaires.append("Le fonctionnement écologique est stable.")
            elif cv_shannon < 0.4:
                commentaires.append("La dynamique est modérément variable.")
            else:
                commentaires.append("Forte variabilité → instabilité écologique.")
        else:
            commentaires.append("Stabilité non calculable sur cette période.")

        if dominance_ratio < 0.40:
            commentaires.append("Le peuplement est relativement bien réparti entre les espèces.")
        elif dominance_ratio < 0.60:
            commentaires.append("Quelques espèces structurent fortement le peuplement.")
        else:
            commentaires.append("Le site est fortement dominé par un petit nombre d'espèces.")

        if prop_nuit_diag < 50:
            commentaires.append("L'activité est majoritairement diurne, ce qui suggère une bonne quiétude.")
        elif prop_nuit_diag < 70:
            commentaires.append("La nocturnité est modérée, avec un possible effet de pression humaine.")
        else:
            commentaires.append("La forte nocturnité peut traduire un évitement de l'activité humaine diurne.")

        st.markdown("### 🧾 Lecture écologique")
        st.write(" ".join(commentaires))

        # ---------------------------------------------------------
        # 8. Forces / Points de vigilance / Recommandations
        # ---------------------------------------------------------
        st.markdown("### 🧭 Forces / Points de vigilance / Recommandations")

        forces = []
        vigilances = []
        recommandations = []

        # Forces
        if score_e1c >= E1C_THRESHOLDS_SOUND["high"]:
            forces.append("Indice E1C excellent dans le référentiel acoustique.")
        elif score_e1c >= E1C_THRESHOLDS_SOUND["medium"]:
            forces.append("Indice E1C bon, cohérent avec un site de bon niveau écologique.")
        if score_shannon >= 60:
            forces.append("Diversité spécifique satisfaisante à élevée.")
        if score_pielou >= 60:
            forces.append("Répartition équilibrée des abondances.")
        if score_simpson >= 60:
            forces.append("Bon nombre effectif d’espèces.")
        if pd.notna(cv_shannon) and cv_shannon < DIAG_THRESHOLDS_SOUND["cv_stable"]:
            forces.append("Bonne stabilité temporelle des indices.")
        if dominance_ratio < DIAG_THRESHOLDS_SOUND["dominance_good"]:
            forces.append("Absence de forte domination par un petit nombre d'espèces.")
        if prop_nuit_diag < DIAG_THRESHOLDS_SOUND["nocturnite_low"]:
            forces.append("Structure horaire majoritairement diurne.")

        # Vigilances
        if score_e1c < E1C_THRESHOLDS_SOUND["low"]:
            vigilances.append("Indice E1C faible dans le référentiel acoustique.")
        if score_shannon < 40:
            vigilances.append("Diversité spécifique limitée.")
        if score_pielou < 40:
            vigilances.append("Répartition inégale des abondances.")
        if score_simpson < 40:
            vigilances.append("Nombre effectif d’espèces faible.")
        if pd.notna(cv_shannon) and cv_shannon >= DIAG_THRESHOLDS_SOUND["cv_medium"]:
            vigilances.append("Variabilité temporelle élevée, traduisant une instabilité possible.")
        if dominance_ratio >= DIAG_THRESHOLDS_SOUND["dominance_medium"]:
            vigilances.append("Forte domination de quelques espèces.")
        if prop_nuit_diag >= DIAG_THRESHOLDS_SOUND["nocturnite_medium"]:
            vigilances.append("Structure horaire fortement nocturne, à interpréter selon les espèces et la saison.")

        # Recommandations
        if score_e1c < E1C_THRESHOLDS_SOUND["high"]:
            recommandations.append("Poursuivre le suivi pour confirmer la trajectoire écologique du site dans le temps.")
        if score_shannon < 40 or score_simpson < 40:
            recommandations.append("Renforcer la diversité des habitats et la connectivité écologique autour du site.")
        if dominance_ratio >= DIAG_THRESHOLDS_SOUND["dominance_medium"]:
            recommandations.append("Examiner les facteurs favorisant la sur-dominance de certaines espèces.")
        if pd.notna(cv_shannon) and cv_shannon >= DIAG_THRESHOLDS_SOUND["cv_medium"]:
            recommandations.append("Analyser les facteurs saisonniers ou de gestion pouvant expliquer l’instabilité observée.")
        if prop_nuit_diag >= DIAG_THRESHOLDS_SOUND["nocturnite_medium"]:
            recommandations.append("Interpréter la structure horaire en lien avec les espèces détectées, la saison et le protocole d’enregistrement.")

        if not forces:
            forces.append("Aucun signal écologique fortement positif ne ressort nettement sur la période considérée.")
        if not vigilances:
            vigilances.append("Pas de point de vigilance majeur détecté sur la période analysée.")
        if not recommandations:
            recommandations.append("Maintenir le protocole de suivi actuel afin de consolider les tendances observées.")

        col_f, col_v, col_r = st.columns(3)

        with col_f:
            st.markdown("#### ✅ Forces")
            for item in forces:
                st.write(f"• {item}")

        with col_v:
            st.markdown("#### ⚠️ Points de vigilance")
            for item in vigilances:
                st.write(f"• {item}")

        with col_r:
            st.markdown("#### 💡 Recommandations")
            for item in recommandations:
                st.write(f"• {item}")

        # ---------------------------------------------------------
        # 9. Définition de l’indice E1C
        # ---------------------------------------------------------
        st.info(
            "L’indice E1C (Every1Counts) combine la diversité (Shannon), "
            "l’équilibre des abondances (Piélou), la structure du peuplement (Simpson 1/D) "
            "et la stabilité temporelle. La calibration SON repose sur les sites acoustiques "
            "de référence intégrés au référentiel Every1Counts."
        )
    # ---------------- TAB EXPORT ----------------
    with tab_export:
        st.subheader("📥 Exploration et Export des données")

        st.markdown("#### 🔍 Rechercher dans le jeu de données")
        search_query = st.text_input("Tapez un mot-clé (nom d'espèce, site, etc.) :", "")

        if search_query:
            df_filtered = df[df.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)]
            st.success(f"✅ {len(df_filtered)} résultat(s) trouvé(s) pour '{search_query}'")
        else:
            df_filtered = df

        st.markdown("#### 📋 Aperçu du Dataset")
        st.write("Affichage des 10 premières lignes :")
        st.dataframe(df_filtered.head(10), use_container_width=True, hide_index=True, height=400)

        st.markdown("#### 💾 Téléchargement")
        csv_data = df_filtered.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Télécharger le dataset complet (CSV)",
            data=csv_data,
            file_name='export_biodiversite_site.csv',
            mime='text/csv',
            help="Cliquez ici pour télécharger les données filtrées au format CSV"
        )

        if search_query:
            st.info("💡 Le bouton de téléchargement s'adapte à votre recherche actuelle.")

else:
    st.warning("⚠️ Aucune donnée pour cette sélection.")