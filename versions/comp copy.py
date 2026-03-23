# comp.py

# =========================================================
# IMPORTS
# =========================================================
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


def _normaliser(x, min_tick, max_tick):
    """Convertit une valeur réelle en position normalisée entre 0 et 1."""
    if max_tick == min_tick:
        return 0.5
    return (x - min_tick) / (max_tick - min_tick)


def generer_graphe_indice(
    val_site,
    nom_indice,
    sites_fixes,
    mode="standard",
    min_tick_force=None,
    max_tick_force=None
):
    """
    Génère un graphique comparatif horizontal de type jauge
    pour positionner un site par rapport à des références fixes.
    Toutes les barres ont la même longueur visuelle, quel que soit l'indice.
    """

    # =========================================================
    # 1. COULEURS / CHARTE
    # =========================================================
    C_BLEU_ETIQUETTE = "#2571A3"
    C_FOND = "#FBF4EC"
    C_VERT_SOMBRE = "#2D4E28"
    C_VIOLET = "#4F479B"

    DEGRADE_HEX = ["#86193F", "#C1B900", "#2D4E28"]

    # =========================================================
    # 2. VALEURS À AFFICHER
    # =========================================================
    valeurs_ref = [site["score"] for site in sites_fixes]
    toutes_valeurs = [val_site] + valeurs_ref

    val_min = min(toutes_valeurs)
    val_max = max(toutes_valeurs)

    # =========================================================
    # 3. BORNES SELON LE MODE
    # =========================================================

    if mode == "standard":
        min_tick = 0
        max_tick = max(3, math.ceil(val_max + 0.2))
        pas_graduation = 1.0

    elif mode == "zoom":
        min_tick = math.floor((val_min - 0.3) * 10) / 10
        max_tick = math.ceil((val_max + 0.3) * 10) / 10
        pas_graduation = 0.5

    elif mode == "expert":
        min_tick = math.floor((val_min - 0.15) * 10) / 10
        max_tick = math.ceil((val_max + 0.15) * 10) / 10
        pas_graduation = 0.2

    elif mode == "large":
        min_tick = math.floor((val_min - 1))
        max_tick = math.ceil((val_max + 1))
        pas_graduation = 5

    elif mode == "tiny":
        min_tick = 0
        max_tick = 1
        pas_graduation = 0.2

    else:
        raise ValueError("mode doit être 'standard', 'zoom' ou 'expert'")

    # =========================================================
    # 4. FORÇAGE + SÉCURITÉ POUR INCLURE LE SITE
    # =========================================================
    if min_tick_force is not None:
        min_tick = min_tick_force

    if max_tick_force is not None:
        max_tick = max_tick_force

    if val_site < min_tick:
        marge = 0.05 * abs(val_site) if val_site != 0 else 0.05
        min_tick = val_site - marge

    if val_site > max_tick:
        marge = 0.05 * abs(val_site) if val_site != 0 else 0.05
        max_tick = val_site + marge

    if max_tick <= min_tick:
        max_tick = min_tick + 1

    plage_valeurs = max_tick - min_tick

    # =========================================================
    # 5. FIGURE
    # =========================================================
    fig, ax = plt.subplots(figsize=(12, 1.9), dpi=100)

    fig.patch.set_facecolor(C_FOND)
    ax.set_facecolor(C_FOND)

    # Coordonnées normalisées :
    # barre de 0 à 1, espace à gauche pour le nom de l'indice
    ax.set_xlim(-0.40, 1.08)
    ax.set_ylim(-0.10, 0.95)
    ax.axis("off")

    # =========================================================
    # 6. DIMENSIONS
    # =========================================================
    H_BLOC_FIXE = 0.18
    LARGEUR_BLOC_FIXE = 0.09
    LARGEUR_BLOC_SITE = 0.09

    EPAISSEUR_BARRE = 0.18
    POSITION_Y = 0.28

    SOMMET_BARRE = POSITION_Y + EPAISSEUR_BARRE
    CENTRE_BARRE = POSITION_Y + EPAISSEUR_BARRE / 2

    # =========================================================
    # 7. NOM DE L'INDICE À GAUCHE
    # =========================================================
    ax.text(
        -0.08,
        CENTRE_BARRE,
        nom_indice,
        fontsize=11,
        fontweight="bold",
        color=C_VERT_SOMBRE,
        ha="right",
        va="center"
    )

    # =========================================================
    # 8. BARRE DE DÉGRADÉ NORMALISÉE
    # =========================================================
    cmap = LinearSegmentedColormap.from_list("biodiv", DEGRADE_HEX)
    n_segments = 300
    largeur_segment = 1.0 / n_segments

    for i in range(n_segments):
        x0 = i * largeur_segment
        ax.add_patch(
            patches.Rectangle(
                (x0, POSITION_Y),
                largeur_segment,
                EPAISSEUR_BARRE,
                color=cmap(i / (n_segments - 1)),
                lw=0,
                zorder=1
            )
        )

    # =========================================================
    # 9. GRADUATIONS
    # =========================================================
    HAUTEUR_TRAIT = 0.08

    Y_DEBUT_TRAIT = POSITION_Y  # démarre au bas de la barre
    Y_FIN_TRAIT = SOMMET_BARRE  # monte jusqu'en haut de la barre

    Y_TEXTE_GRAD = POSITION_Y - 0.15

    n_grad = int(round((max_tick - min_tick) / pas_graduation)) + 1
    graduations = [round(min_tick + i * pas_graduation, 10) for i in range(n_grad)]

    # Garde uniquement les graduations dans la plage
    graduations = [g for g in graduations if min_tick - 1e-9 <= g <= max_tick + 1e-9]

    for val in graduations:
        xg = _normaliser(val, min_tick, max_tick)

        ax.plot(
            [xg, xg],
            [Y_DEBUT_TRAIT, Y_FIN_TRAIT],
            color="black",
            lw=1,
            zorder=2
        )

        if abs(val - round(val)) < 1e-9:
            etiquette = str(int(round(val)))
        else:
            etiquette = f"{val:.1f}"

        ax.text(
            xg,
            Y_TEXTE_GRAD,
            etiquette,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            zorder=2
        )

    # =========================================================
    # 10. BLOCS FIXES BLEUS
    # =========================================================
    DISTANCE_BLOC_BARRE = 0.06
    Y_BLOC_FIXE = SOMMET_BARRE + DISTANCE_BLOC_BARRE
    Y_TEXTE_FIXE = Y_BLOC_FIXE + H_BLOC_FIXE / 2
    DEMI_BASE_FLECHE = 0.010

    for site in sites_fixes:
        val = site["score"]
        x = _normaliser(val, min_tick, max_tick)

        ax.add_patch(
            patches.FancyBboxPatch(
                (x - LARGEUR_BLOC_FIXE / 2, Y_BLOC_FIXE),
                LARGEUR_BLOC_FIXE,
                H_BLOC_FIXE,
                boxstyle="round,pad=0.02",
                facecolor=C_BLEU_ETIQUETTE,
                edgecolor="none",
                zorder=4
            )
        )

        ax.text(
            x,
            Y_TEXTE_FIXE,
            f"{site['label']}\n{val:.2f}",
            color="white",
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            zorder=5
        )

        triangle = patches.Polygon(
            [
                (x - DEMI_BASE_FLECHE, Y_BLOC_FIXE),
                (x + DEMI_BASE_FLECHE, Y_BLOC_FIXE),
                (x, SOMMET_BARRE)
            ],
            closed=True,
            facecolor=C_BLEU_ETIQUETTE,
            edgecolor="none",
            zorder=4
        )
        ax.add_patch(triangle)

    # =========================================================
    # 11. BLOC SITE
    # =========================================================
    x_site = _normaliser(val_site, min_tick, max_tick)

    # garde le bloc légèrement dans la zone visible
    x_site = min(max(x_site, LARGEUR_BLOC_SITE / 2), 1 - LARGEUR_BLOC_SITE / 2)

    ax.add_patch(
        patches.Rectangle(
            (x_site - LARGEUR_BLOC_SITE / 2, POSITION_Y),
            LARGEUR_BLOC_SITE,
            EPAISSEUR_BARRE,
            facecolor=C_VIOLET,
            edgecolor="white",
            lw=1.2,
            zorder=10
        )
    )

    ax.text(
        x_site,
        CENTRE_BARRE,
        f"SITE\n{val_site:.2f}",
        color="white",
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        zorder=11
    )

    return fig


def generer_graphe_shannon(
    val_site,
    nom_indice,
    sites_fixes,
    mode="standard",
    min_tick_force=None,
    max_tick_force=None
):
    return generer_graphe_indice(
        val_site=val_site,
        nom_indice=nom_indice,
        sites_fixes=sites_fixes,
        mode=mode,
        min_tick_force=min_tick_force,
        max_tick_force=max_tick_force
    )