#  Diamond Cut Prediction

Prédiction de la qualité de coupe des diamants (cut) à partir de leurs caractéristiques physiques – un projet de machine learning structuré de manière professionnelle.

---

##  Objectif

L’objectif de ce projet est de développer un modèle de machine learning capable de prédire la **qualité de coupe** (`cut`) d’un diamant en fonction de ses propriétés physiques comme le poids (`carat`), la couleur ou blancheur (`color`), la clarté ou pureté (`clarity`), la profondeur (`depth`), la largeur de la table (`table`), la longueur(`x`), la largeur(`y`), la profondeur(`z`).

---

##  Contexte métier

Ce projet s’inspire d’un **contexte professionnel plausible** : un **revendeur de diamants en ligne** souhaite automatiser l'évaluation de la qualité des diamants afin de :

- **Améliorer la cohérence** de ses descriptions produits,
- **Réduire la subjectivité** dans l’évaluation humaine,
- **Accélérer le processus de mise en vente**.

L’enjeu est donc **opérationnel et économique** : un mauvais classement de la coupe peut impacter le prix, la perception client, voire les retours.

---

##  Données

Le jeu de données provient de la base `diamonds` de Kaggle. Il contient plus de 50 000 lignes et inclut les variables suivantes :

- `carat`, `depth`, `table`, `x`, `y`, `z` : variables numériques
- `color`, `clarity`, `cut` : variables catégorielles

L’étiquette cible est la variable `cut` (ordinale : Fair < Good < Very Good < Premium < Ideal).

---

##  Métriques & Contraintes

Le modèle doit prendre en compte que :

- **La variable cible est ordinale**, pas purement catégorielle.
- Toutes les erreurs ne se valent pas : confondre *Ideal* avec *Premium* est moins grave que *Ideal* avec *Fair*.

C’est pourquoi les métriques sélectionnées sont :

- **F1-score pondéré --> 80% minimum** : équilibre entre précision et rappel, en tenant compte du déséquilibre des classes. 
- **Confusion matrix** : pour visualiser où les erreurs se concentrent.
- **Cohen’s Kappa / Quadratic Weighted Kappa --> 75% minimum** : pour évaluer la justesse sur des classes ordinales.

---

##  Organisation du projet

```bash
diamond_cut_prediction/
│
├── README.md               # Ce fichier
├── requirements.txt        # Librairies nécessaires
├── data/                   # Données brutes et nettoyées
├── notebooks/              # Jupyter notebooks (EDA, modèle, etc.)
├── src/                    # Code source : loading, features, training
├── scripts/                # Scripts exécutables (CLI)
├── config/                 # Paramètres YAML et JSON
├── models/                 # Fichiers de modèles entraînés
├── outputs/                # Logs, graphes, métriques
└── tests/                  # Tests unitaires
