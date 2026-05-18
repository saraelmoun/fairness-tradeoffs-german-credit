# PACR-AP — Plausible Actionable Counterfactual Recourse with Action Paths

Notebook autonome appliqué au jeu **German Credit** (OpenML).
Cours **IADATA708**, 2025–2026.

## Contenu

```
.
├── README.md                  # ce fichier
├── requirements.txt           # dépendances Python
├── pacr_ap.ipynb              # notebook principal — l'analyse complète
├── pacr_ap.py                 # module Python — toute la mécanique de la méthode
├── schema-actions.yaml        # politique d'actionnabilité (3 variants)
└── figures/                   # diagrammes d'overview du §2
    ├── pacr_ap_pipeline.png
    └── pacr_ap_audit.png
```

## Installation

Python 3.10+ recommandé.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer le notebook

```bash
jupyter lab pacr_ap.ipynb
```

Puis **Run All**. Le notebook télécharge automatiquement German Credit depuis
OpenML à la première exécution.

Temps de calcul indicatif : **5–10 minutes** sur une machine récente (la majorité
est consommée par les 30 individus × graphe BFS × bootstrap robustness, et par
le triple stress-test du §7).

## Vue d'ensemble

PACR-AP répond à : *« pour un emprunteur refusé, existe-t-il une séquence
d'actions plausibles qui basculerait la décision ? et cette possibilité est-elle
équitable entre groupes d'âge ? »*

Le notebook suit le plan :

1. La méthode en bref + emprunts à la littérature (Wachter, Ustun, FACE, DiCE)
2. Overview du pipeline (deux diagrammes)
3. Setup expérimental
4. Application sur un individu refusé — démo concrète
5. Audit fairness par groupe (jeunes vs adultes)
6. Exploration interactive (dashboard Altair)
7. Robustesse du verdict au schéma d'actions
8. Conclusion et limites

Trois annexes en fin de notebook : (A) construction pas-à-pas de la méthode,
(B) littérature détaillée, (C) mining des magnitudes depuis le training.

## Reproductibilité

Tous les seeds sont fixés (`SEED = 42`). Le split, le mining d'actions et le
bootstrap sont déterministes. Les résultats du notebook sont reproductibles à
l'identique sur la même version de scikit-learn.
