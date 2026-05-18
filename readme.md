# When Machine Learning Fails — Online Shoppers

> **École Centrale Casablanca · Spring 2026 · Introduction to AI and Machine Learning**
> Mini-project: *Diagnosing and Repairing Failure Modes of a Non-Linear Model*

**Authors / Auteures :** Samia Touile · Inass Semmar
**Instructors / Encadrement :** Kawtar Zerhouni & Rym Nassih (UTER MID@S)

---

## Project Overview / Présentation

Ce projet investigue un **mode d'échec** d'un modèle de Machine Learning non-linéaire sur le dataset [Online Shoppers Intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) (UCI 468). Conformément à la démarche imposée par le sujet, nous adoptons une posture **scientifique plutôt qu'ingénierique** :

1. Forcer le modèle à échouer.
2. Comprendre **pourquoi** il échoue (hypothèse causale falsifiable + contrôle bidirectionnel).
3. **Réparer** la cause (et non le symptôme).
4. **Discuter** les menaces à la validité des conclusions.

**Failure mode investigué :** *Overfitting and generalization gap* (§5.2 du sujet).

**Modèle :** `GradientBoostingClassifier` (scikit-learn, non-linéaire imposé par §3 du sujet).

---

## Key Result / Résultat clé

> La **capacité du modèle** (profondeur, nombre d'arbres, granularité des feuilles) est la cause causale de l'écart train-validation observé. Une régularisation **guidée par diagnostic** (correction E') réduit cet écart de **72.3 %** *sans dégrader significativement* le F1 test (t = 1.639, p = 0.140).

| Métrique | Référence (A) | Correction (E') | Effet |
|----------|---------------|------------------|-------|
| Gap F1 (train − test) | 0.125 | **0.035** | **−72.3 %** |
| F1 test (purchase) | 0.649 ± 0.005 | 0.641 ± 0.008 | Non significatif (p = 0.140) |
| ROC-AUC test | 0.927 | 0.925 | Stable |

*Moyennes sur 5 seeds (42–46). Source autoritative : `report/experiment_results.json`.*

---

## Repository Structure / Structure du dépôt

```
.
├── README.md                                # Ce fichier
├── requirements.txt                         # Dépendances Python
├── .gitignore
├── Rapport_TouileSemmar_WhenMLfails.pdf     # Rapport final compilé (~16 pages)
│
├── data/
│   └── online_shoppers_intention.csv        # Dataset (12 330 sessions, 17 features)
│
├── 01_eda.ipynb                             # Pipeline "cassé" : EDA + modèle référence
│                                              ├─ Class imbalance et effet temporel
│                                              ├─ Modèle de référence (GBM standard)
│                                              ├─ Symptômes : gap, F1 par mois, learning curve
│                                              ├─ Réfutation explicite d'un shortcut sur Month
│                                              └─ Sauvegarde : report/reference_metrics.json
│
├── 02_experiment.ipynb                      # Pipeline "corrigé" : expérience contrôlée
│                                              ├─ Symptôme quantifié (gap ≈ 0.13)
│                                              ├─ Contrôle bidirectionnel (gap × 9)
│                                              ├─ Validation curves → diagnostic causal
│                                              ├─ 5 corrections + critère a priori (toutes validées)
│                                              ├─ Correction officielle E' (min du gap)
│                                              └─ t-test, learning curve corrigée
│
└── report/                                  # Figures et métriques générées
    ├── fig01_target_distribution.png
    ├── fig02_month_effect.png
    ├── fig03_confusion_ref.png
    ├── fig04_feature_importances.png
    ├── fig05_perf_par_mois.png
    ├── fig06_learning_curve.png
    ├── fig07_correlation_matrix.png
    ├── fig11_learning_curves_detailed.png
    ├── fig11bis_bidirectional_control.png
    ├── fig12_validation_curves.png
    ├── fig13_corrections_comparison.png    # 6 modèles : A, B, C, D, E, E'
    ├── fig14_learning_curve_corrected.png
    ├── reference_metrics.json              # Source autoritative NB 01
    └── experiment_results.json             # Source autoritative NB 02
```

---

## Installation / Installation

### Prérequis / Prerequisites

- Python 3.12 (testé sur cette version)
- pip ou conda
- ~500 MB d'espace disque pour les dépendances

### Étapes / Steps

```bash
# 1. Cloner le dépôt
git clone https://github.com/Samiatouile/ml-fails-online-shoppers.git
cd ml-fails-online-shoppers

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv

# Activer le venv :
# Linux/Mac :
source venv/bin/activate
# Windows PowerShell :
.\venv\Scripts\Activate.ps1

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter
jupyter lab
# OU ouvrir directement dans VS Code
code .
```

### Dépendances principales / Main dependencies

Versions exactes dans `requirements.txt`. Principales :

```
pandas==3.0.3
numpy==2.4.4
scikit-learn==1.8.0
scipy==1.17.1
matplotlib==3.10.9
seaborn==0.13.2
jupyter==1.1.1
```

---

## Usage / Utilisation

### Reproduire les résultats du rapport

Les deux notebooks sont **exécutables indépendamment** (conformément à §8.1 du sujet) : le notebook 02 recharge le CSV et refait le preprocessing complet, sans dépendre du notebook 01.

```bash
# Option 1 : Exécution séquentielle (dans Jupyter)
# Ouvrir 01_eda.ipynb        → Kernel → Restart & Run All
# Ouvrir 02_experiment.ipynb → Kernel → Restart & Run All

# Option 2 : Exécution en ligne de commande
jupyter nbconvert --to notebook --execute 01_eda.ipynb
jupyter nbconvert --to notebook --execute 02_experiment.ipynb
```

**Temps d'exécution estimé :**
- `01_eda.ipynb` : ~2 minutes
- `02_experiment.ipynb` : ~8-12 minutes (entraînement de 6 modèles × 5 seeds)

### Reproductibilité / Reproducibility

- **Seed principale :** `RANDOM_STATE = 42`
- **Variance mesurée sur 5 seeds :** `42, 43, 44, 45, 46`
- **Split :** stratifié 80/20 sur la cible `Revenue`
- `n_jobs=1` sur `learning_curve`, `validation_curve`, `cross_val_score` pour garantir le déterminisme strict
- La validation croisée est faite **uniquement sur le train** ; le test set reste isolé
- Sources autoritatives des chiffres : `report/reference_metrics.json` et `report/experiment_results.json`

---

## Investigation Pipeline / Démarche d'investigation

La démarche suit la structure imposée par le §6 du sujet :

### 1. Research Question (§6.1)

> Le `GradientBoostingClassifier` entraîné sur Online Shoppers présente-t-il un overfitting structurel, tel qu'une réduction ciblée de sa capacité — guidée par le diagnostic des validation curves — réduise l'écart train-validation d'au moins 30 % sans dégrader significativement le F1 test ?

### 2. Observed Symptom (§6.2)

- F1 train ≈ 0.774 vs F1 test ≈ 0.645 → **gap = 0.129**
- Train F1 ≈ 0.95 sur petit jeu (n ≈ 1 000) → **mémorisation quasi-parfaite**
- F1 par mois hétérogène : 0.48 (Jul/Aug) à 0.77 (Mar)
- Réfutation d'un shortcut sur `Month` : importance = 5.4 % vs PageValues = 66.9 % (ratio 12.3×)

### 3. Causal Hypothesis & Control (§6.3)

- **H1** : la capacité du modèle est la cause du gap.
- **Contrôle bidirectionnel** : low / reference / high capacity → gaps de **0.039 / 0.129 / 0.351** (**monotonie stricte, ratio × 9**)
- **Validation curves** : `max_depth` identifié comme principal vecteur (× 13 entre depth=2 et depth=10).

### 4. Correction (§6.4)

5 corrections testées contre un **critère de succès défini a priori** :
- **(C1)** Réduction du gap ≥ 30 % (gap ≤ 0.090)
- **(C2)** F1 test ≥ référence − 1 σ_CV (≈ 0.626)

**Résultat : les 5 corrections (B, C, D, E, E') valident toutes le critère**, ce qui constitue un indicateur de robustesse du diagnostic causal (le gap se réduit quel que soit le levier choisi parmi ceux dérivés de l'analyse de capacité).

**Correction officielle retenue : E'** — sélectionnée comme celle qui **minimise le gap** parmi les validées (attaque maximale de la cause, pas optimisation du F1) :
- `max_depth=2, min_samples_leaf=100, learning_rate=0.03, subsample=0.7, early_stopping=on`
- Chaque hyperparamètre est traçable à une observation chiffrée des validation curves.

### 5. Validation

- Gap réduit de **72.3 %** (0.125 → 0.035)
- F1 test maintenu : Δ = −0.008, t = 1.639, **p = 0.140** (t-test sur 5 seeds, différence non significative)
- Learning curve corrigée : courbes train et validation quasi-confondues sur tout l'intervalle.

---

## Report / Rapport

Le rapport final (~16 pages, en français) est inclus dans le dépôt :

📄 **`Rapport_TouileSemmar_WhenMLfails.pdf`**

Et a été déposé sur edunao + envoyé par mail aux encadrantes.

Structure (conforme au §8.2 du sujet) :
1. Introduction — dataset, modèle, choix du failure mode
2. Question de recherche (§6.1)
3. Symptôme observé (§6.2)
4. Hypothèse causale et expérience contrôlée (§6.3)
5. Correction proposée et évaluation (§6.4)
6. Menaces à la validité (§7)
7. Reproductibilité (§8)
8. Conclusion et apprentissages

---

## Limitations / Limites identifiées

Conformément à l'exigence §7 du sujet, les principales menaces à la validité de nos conclusions sont :

1. **Non-significativité statistique** : avec n = 5 seeds, p = 0.140 ne permet pas de **prouver** l'équivalence F1(A) ≈ F1(E') — seulement de constater l'absence de différence détectable. Un échantillon plus grand permettrait de mieux trancher.

2. **Caractère borderline de l'overfitting initial** (gap = 0.129, zone modérée [0.10 ; 0.15]) : les conclusions ne se transposent pas mécaniquement à des configurations plus extrêmes (dataset plus petit, modèle plus capacitaire).

3. **Plafond imposé par le déséquilibre non traité** : le déséquilibre 84/16 — identifié comme le failure mode dominant — n'a pas été traité ici. La marge de progression sur le F1 (classe Achat) est donc intrinsèquement bornée.

4. **Confondants potentiels dans E'** : 5 hyperparamètres modifiés simultanément. Le confounding est *atténué* par les corrections mono-axes (B, C, D) qui valident chacune indépendamment, mais B (`depth=2` seul) atteint un gap de 0.040 très proche d'E' (0.035) — la non-trivialité d'E' est *suggérée mais non strictement démontrée*.

5. **Choix de la métrique de gap** : `train_F1 − test_F1`. Le gap d'AUC est seulement de 0.035, ce qui suggère qu'une partie de l'"overfitting" mesuré pourrait être un problème de calibration du seuil de décision.

6. **Absence de validation hors-distribution** : toutes les évaluations sont sur le même split aléatoire. Un split temporel renforcerait la démonstration de meilleure généralisation.

Voir §6 du rapport pour la discussion complète.

---

## License & Acknowledgments

Project carried out as part of the *Introduction to AI and Machine Learning* course at École Centrale Casablanca (Spring 2026).

**Dataset citation :**
> Sakar, C.O., Polat, S.O., Katircioglu, M. *et al.* "Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks." *Neural Comput & Applic* 31, 6893–6908 (2019). [UCI ML Repository link](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

**Course material :** © UTER MID@S — École Centrale Casablanca.