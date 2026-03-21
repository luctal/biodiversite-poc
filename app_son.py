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

# --- CONFIGURATION DES RÉFÉRENCES ÉCOLOGIQUES ---

REFERENCES_INDICES = {
    "Shannon": {
        "nom": "Indice de Shannon",
        "mode": "standard",
        "min_tick": 2,
        "max_tick": 4,
        "sites": [
            {"label": "ET", "score": 3.35, "err": 0.02, "desc": "Etrechy, zone périurbaine"},
            {"label": "LV", "score": 3.29, "err": 0.02, "desc": "Lavallière, parc d’hôtel"},
            {"label": "LP", "score": 3.32, "err": 0.02, "desc": "La Peyruche, vignoble bio"}
        ]
    },
    "Richesse": {
        "nom": "Richesse spécifique",
        "mode": "large",
        "min_tick": 80,
        "max_tick": 120,
        "sites": [
            {"label": "ET", "score": 92, "err": 3, "desc": "Etrechy, zone périurbaine"},
            {"label": "LV", "score": 87, "err": 3, "desc": "Lavallière, parc d’hôtel"},
            {"label": "LP", "score": 108, "err": 4, "desc": "La Peyruche, vignoble bio"}
        ]
    },
    "InvD": {
        "nom": "Nombre effectif (1 / D)",
        "mode": "standard",
        "min_tick": 5,
        "max_tick": 15,
        "sites": [
            {"label": "ET", "score": 17.4, "err": 0.4, "desc": "Etrechy, zone périurbaine"},
            {"label": "LV", "score": 16.6, "err": 0.4, "desc": "Lavallière, parc d’hôtel"},
            {"label": "LP", "score": 12.6, "err": 0.4, "desc": "La Peyruche, vignoble bio"}
        ]
    },
    "Pielou": {
        "nom": "Équitabilité de Piélou",
        "mode": "tiny",
        "min_tick": 0,
        "max_tick": 1,
        "sites": [
            {"label": "ET", "score": 0.74, "err": 0.01, "desc": "Etrechy, zone périurbaine"},
            {"label": "LV", "score": 0.74, "err": 0.01, "desc": "Lavallière, parc d’hôtel"},
            {"label": "LP", "score": 0.71, "err": 0.01, "desc": "La Peyruche, vignoble bio"}
        ]
    },
"IAJC": {
    "nom": "Indice d'Activité (IAJC)",
    "mode": "large",
    "min_tick": 5,
    "max_tick": 25,
    "sites": [
        {"label": "ET", "score": 20.1, "err": 0.0},
        {"label": "LV", "score": 12.5, "err": 0.0},
        {"label": "LP", "score": 8.4, "err": 0.0}
    ]
}
}


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
    df = pd.read_csv(uploaded_file, sep=None, engine='python')

    mapping = {
        'Nom vernaculaire': 'vernacular_name',
        'Common Name': 'vernacular_name',
        'Common_name': 'vernacular_name',
        'vernacular_name': 'vernacular_name',

        'Nom scientifique': 'scientific_name',
        'Scientific name': 'scientific_name',
        'Scientifique_name': 'scientific_name',
        'scientific_name': 'scientific_name',

        'Hotspot': 'site',
        'Site': 'site',
        'site': 'site',

        'Indice de confiance BirdNet': 'Birdnet_confidence_index',
        'Birdnet_confidence_index': 'Birdnet_confidence_index'
    }

    df = df.rename(columns=mapping)

    if 'vernacular_name' not in df.columns:
        st.error(f"⚠️ Impossible de trouver la colonne des espèces. Colonnes présentes : {list(df.columns)}")
        st.info("💡 Assurez-vous que votre fichier contient une colonne nommée 'Common Name', 'Nom vernaculaire' ou 'vernacular_name'.")
        st.stop()

    if 'startdate' not in df.columns:
        st.error("⚠️ Impossible de trouver la colonne 'startdate'.")
        st.stop()

    if 'detection_count' not in df.columns:
        df['detection_count'] = 1

    # Forcer detection_count en numérique
    df['detection_count'] = pd.to_numeric(df['detection_count'], errors='coerce')

    # Remplacer les valeurs invalides par 1
    df['detection_count'] = df['detection_count'].fillna(1)

    # Forcer latitude / longitude en numérique si présentes
    if 'latitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')

    if 'longitude' in df.columns:
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    df['startdate'] = pd.to_datetime(df['startdate'], errors='coerce')
    df = df.dropna(subset=['startdate'])
    df['Heure'] = df['startdate'].dt.hour

    return df


@st.cache_data
def load_comparison_data():
    path = "datasets/20260303-indices-sites.csv"
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# 3. SIDEBAR - CHARGEMENT
st.sidebar.title("📁 Données")
uploaded_file = st.sidebar.file_uploader("Charger un dataset CSV", type=["csv"])

if uploaded_file is not None:
    raw_df = load_data(uploaded_file)
    df_bench = load_comparison_data()
else:
    st.info("👋 Veuillez charger un fichier CSV dans la barre latérale pour commencer l'analyse.")
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

def bootstrap_iajc(data, n_iterations=400):
    """
    Ancienne version de l'IAJC :
    IAJC = nombre total de détections / (nombre de jours * nombre de sites)

    Remarque :
    - ici, 'site' est utilisé comme proxy du nombre de caméras / hotspots
    - l'effort est supposé constant sur toute la période
    """
    if data.empty:
        return 0, 0

    n_jours = (data['startdate'].max() - data['startdate'].min()).days + 1
    n_cameras = data['site'].nunique() if 'site' in data.columns else 1
    effort = n_jours * n_cameras if n_jours > 0 and n_cameras > 0 else 1

    iajc_sims = []

    for _ in range(n_iterations):
        sample = data.sample(frac=1.0, replace=True)

        # Si detection_count existe, on l'utilise.
        # Sinon, chaque ligne compte pour 1 observation.
        if 'detection_count' in sample.columns:
            total_animaux = sample['detection_count'].sum()
        else:
            total_animaux = len(sample)

        iajc_sims.append(total_animaux / effort)

    return np.mean(iajc_sims), np.std(iajc_sims)


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
    # 1. Préparation des colonnes temporelles
    # ---------------------------------------------------------
    df_temp = df_input.copy()
    df_temp['Semaine'] = df_temp['startdate'].dt.isocalendar().week
    df_temp['Annee'] = df_temp['startdate'].dt.year

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
    # Pour chaque couple (site, année, semaine),
    # on calcule Shannon à partir des abondances d'espèces.
    # ---------------------------------------------------------
    df_stats = df_temp.groupby(['site', 'Annee', 'Semaine']).apply(
        lambda x: calc_shannon(x['vernacular_name'].value_counts())
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
    df_temp = df_input.copy()
    df_temp['Semaine'] = df_temp['startdate'].dt.isocalendar().week
    df_temp['Annee'] = df_temp['startdate'].dt.year

    # ---------------------------------------------------------
    # 2. Calcul de Shannon par site / année / semaine
    # ---------------------------------------------------------
    # Ici on repart des abondances d'espèces à l'intérieur de chaque semaine
    # pour obtenir une distribution hebdomadaire de l'indice de Shannon.
    # ---------------------------------------------------------
    df_stats = df_temp.groupby(['site', 'Annee', 'Semaine']).apply(
        lambda x: -np.sum(
            (x['vernacular_name'].value_counts() / len(x)) *
            np.log(x['vernacular_name'].value_counts() / len(x) + 1e-9)
        )
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

    df['Espèce Graphique'] = df['vernacular_name'].apply(lambda x: x if x in top9 else "Autres")
    df['Semaine'] = df['startdate'].dt.to_period('W').apply(lambda r: r.start_time)

    st.title("🍃 Tableaux de bord - Son")

    tab_global, tab_comparaison, tab_stats, tab_diagnostic, tab_export = st.tabs(
        ["📊 Dashboard Global", "🔬 Comparaison de Sites", "📈 Statistiques", "🧠 Diagnostic écologique", "📥 Export"]
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

        df_weekly_stats = df.groupby(['site', 'Semaine', 'vernacular_name'])['detection_count'].sum().reset_index()

        def compute_shannon(group):
            total = group['detection_count'].sum()
            if total == 0:
                return 0
            p_i = group['detection_count'] / total
            return -1 * (p_i * np.log(p_i + 1e-9)).sum()

        df_dist_shannon = df_weekly_stats.groupby(['site', 'Semaine']).apply(
            compute_shannon, include_groups=False
        ).reset_index(name='shannon_val')

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

    # ---------------- TAB DIAGNOSTIC ----------------
    with tab_diagnostic:
        st.subheader("🧠 Diagnostic écologique")

        # ---------------------------------------------------------
        # 1. Calcul du Shannon hebdomadaire (stabilité)
        # ---------------------------------------------------------
        df_dist_shannon_diag = compute_weekly_shannon_distribution(df)

        # ---------------------------------------------------------
        # 2. Calcul de l’indice E1C
        # ---------------------------------------------------------
        score_e1c, cv_shannon, score_stabilite = compute_indice_e1c(
            bootstrap_results,
            df_dist_shannon_diag
        )

        # ---------------------------------------------------------
        # 3. Affichage des indicateurs principaux
        # ---------------------------------------------------------
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Indice E1C",
                f"{score_e1c:.0f} / 100"
            )

        with c2:
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

        with c3:
            st.metric(
                "Score de stabilité",
                f"{score_stabilite:.2f}"
            )

        # ---------------------------------------------------------
        # 4. Interprétation globale
        # ---------------------------------------------------------
        if score_e1c >= 70:
            st.success("🟢 Indice E1C élevé : écosystème équilibré et fonctionnel.")
        elif score_e1c >= 50:
            st.warning("🟠 Indice E1C intermédiaire : écosystème fonctionnel mais perfectible.")
        else:
            st.error("🔴 Indice E1C faible : écosystème sous pression ou déséquilibré.")

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
        if score_e1c >= 70:
            forces.append("Indice E1C élevé, traduisant un fonctionnement écologique globalement favorable.")
        if bootstrap_results and bootstrap_results['H'][0] >= 1.8:
            forces.append("Diversité spécifique élevée sur la période analysée.")
        if bootstrap_results and bootstrap_results['J'][0] >= 0.6:
            forces.append("Répartition équilibrée des abondances entre les espèces.")
        if pd.notna(cv_shannon) and cv_shannon < 0.2:
            forces.append("Bonne stabilité temporelle des indices de biodiversité.")
        if dominance_ratio < 0.40:
            forces.append("Absence de forte domination par un petit nombre d'espèces.")
        if prop_nuit_diag < 50:
            forces.append("Activité majoritairement diurne, compatible avec une bonne quiétude du site.")

        # Vigilances
        if score_e1c < 50:
            vigilances.append("Indice E1C faible, suggérant un site sous pression ou déséquilibré.")
        if bootstrap_results and bootstrap_results['H'][0] < 1.3:
            vigilances.append("Diversité spécifique limitée.")
        if bootstrap_results and bootstrap_results['J'][0] < 0.4:
            vigilances.append("Répartition très inégale des abondances, avec domination marquée de certaines espèces.")
        if pd.notna(cv_shannon) and cv_shannon >= 0.4:
            vigilances.append("Variabilité temporelle élevée, traduisant une instabilité écologique possible.")
        if dominance_ratio >= 0.60:
            vigilances.append("Forte domination de quelques espèces au sein du peuplement.")
        if prop_nuit_diag >= 70:
            vigilances.append("Nocturnité élevée pouvant traduire un évitement de l'activité humaine diurne.")

        # Recommandations
        if score_e1c < 70:
            recommandations.append(
                "Poursuivre le suivi pour confirmer la trajectoire écologique du site dans le temps.")
        if dominance_ratio >= 0.60:
            recommandations.append(
                "Examiner les conditions de milieu favorisant la sur-dominance de certaines espèces.")
        if pd.notna(cv_shannon) and cv_shannon >= 0.4:
            recommandations.append(
                "Analyser les facteurs saisonniers ou de gestion pouvant expliquer l'instabilité observée.")
        if prop_nuit_diag >= 70:
            recommandations.append(
                "Évaluer les sources potentielles de dérangement diurne : fréquentation, bruit, travaux, circulation.")
        if bootstrap_results and bootstrap_results['H'][0] < 1.3:
            recommandations.append(
                "Renforcer la diversité des habitats et la connectivité écologique à l'échelle du site.")

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
            "l’équilibre des abondances (Piélou) et la stabilité temporelle "
            "pour fournir une lecture synthétique de la qualité écologique du site."
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