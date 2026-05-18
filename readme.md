# When Machine Learning Fails — Online Shoppers

> **École Centrale Casablanca · Spring 2026 · Introduction to AI and Machine Learning**
> Mini-project: *Diagnosing and Repairing Failure Modes of a Non-Linear Model*

**Authors / Auteures :** Samia Touile · Inass Semmar
**Instructors / Encadrement :** Kawtar Zerhouni & Rym Nassih (UTER MID@S)

---

##  Project Overview / Présentation

Ce projet investigue un **mode d'échec** d'un modèle de Machine Learning non-linéaire sur le dataset [Online Shoppers Intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) (UCI 468). Conformément à la démarche imposée par le sujet, nous adoptons une posture **scientifique plutôt qu'ingénierique** :

1. Forcer le modèle à échouer.
2. Comprendre **pourquoi** il échoue (hypothèse causale falsifiable + contrôle).
3. **Réparer** la cause (et non le symptôme).
4. **Discuter** les menaces à la validité des conclusions.

**Failure mode investigué :** *Overfitting and generalization gap* (§5.2 du sujet).

**Modèle :** `GradientBoostingClassifier` (scikit-learn).

---

##  Key Result / Résultat clé

> La **capacité du modèle** (profondeur, nombre d'arbres, granularité des feuilles) est la cause causale de l'écart train-validation observé. Une régularisation **guidée par diagnostic** (correction E$'$) réduit cet écart de **72.9 %** *sans dégrader significativement* le F1 test (p = 0.13).

| Métrique | Référence (A) | Correction (E') | Effet |
|----------|---------------|------------------|-------|
| Gap F1 (train − test) | 0.128 | **0.035** | **−72.9 %** |
| F1 test (purchase) | 0.6509 ± 0.008 | 0.6414 ± 0.008 | Non significatif (p = 0.13) |
| ROC-AUC | 0.9268 | 0.9246 | Stable |

---

##  Repository Structure / Structure du dépôt

```
.
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── data/
│   └── online_shoppers_intention.csv  # Dataset (12 330 sessions, 17 features)
│
├── 01_eda.ipynb                       # Pipeline "cassé" : EDA + modèle référence
│                                        ├─ Class imbalance et effet temporel
│                                        ├─ Modèle de référence (GBM standard)
│                                        ├─ Symptômes : gap, F1 par mois, learning curve
│                                        └─ Sauvegarde : report/reference_metrics.json
│
├── 02_experiment.ipynb                # Pipeline "corrigé" : expérience contrôlée
│                                        ├─ Symptôme quantifié (gap = 0.128)
│                                        ├─ Contrôle bidirectionnel (gap × 9)
│                                        ├─ Validation curves → diagnostic
│                                        ├─ 5 corrections candidates + critère a priori
│                                        ├─ Correction E' (réduction du gap de 72.9 %)
│                                        └─ t-test, learning curve corrigée
│
└── report/                            # Figures et métriques générées
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
    ├── fig13_corrections_comparison.png
    ├── fig14_learning_curve_corrected.png
    ├── reference_metrics.json
    └── corrected_metrics.json
```

---

## Installation / Installation

### Prérequis / Prerequisites

- Python ≥ 3.10 (testé sur 3.12)
- pip ou conda
- ~500 MB d'espace disque pour les dépendances

### Étapes / Steps

```bash
# 1. Cloner le dépôt
git clone https://github.com/<USERNAME>/<REPO_NAME>.git
cd <REPO_NAME>

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

```
pandas        >= 2.0
numpy         >= 1.24
scikit-learn  >= 1.4
matplotlib    >= 3.7
seaborn       >= 0.13
scipy         >= 1.11
jupyter       >= 1.0
```

---

##  Usage / Utilisation

### Reproduire les résultats du rapport

Les deux notebooks sont **exécutables indépendamment** (conformément à §8.1 du sujet). Pour reproduire l'intégralité des résultats :

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
- Les outputs des deux notebooks sont **déterministes** sur ces seeds.

---

##  Investigation Pipeline / Démarche d'investigation

La démarche suit la structure imposée par le §6 du sujet :

### 1. Research Question (§6.1)

> Le `GradientBoostingClassifier` entraîné sur Online Shoppers présente-t-il un overfitting structurel, tel qu'une réduction ciblée de sa capacité — guidée par le diagnostic des validation curves — réduise l'écart train-validation d'au moins 30 % sans dégrader significativement le F1 test ?

### 2. Observed Symptom (§6.2)

- F1 train = 0.7734 vs F1 test = 0.6458 → **gap = 0.128**
- Train F1 = 1.000 sur petit jeu (n = 1 000) → **mémorisation parfaite**
- F1 par mois hétérogène : 0.48 (Jul) à 0.77 (Mar)

### 3. Causal Hypothesis & Control (§6.3)

- **H1** : la capacité du modèle est la cause du gap.
- **Contrôle bidirectionnel** : low / reference / high capacity → gaps de 0.039 / 0.128 / 0.357 (**monotonie stricte, × 9**)
- **Validation curves** : `max_depth` identifié comme principal vecteur (× 13 entre depth=2 et depth=10).

### 4. Correction (§6.4)

5 corrections testées contre un **critère de succès défini a priori** :
- (C1) Réduction du gap ≥ 30 %
- (C2) F1 test ≥ référence − 1 σ_CV

**Correction officielle retenue : E'** (régularisation guidée par diagnostic)
- `max_depth=2, min_samples_leaf=100, learning_rate=0.03, subsample=0.7, early_stopping=on`

### 5. Validation

- Gap réduit de **72.9 %**
- F1 test maintenu (p = 0.13, t-test sur 5 seeds → différence non significative)
- Learning curve corrigée : courbes train et validation quasi-confondues.

---

##  Report / Rapport

Le rapport complet (~14 pages, en français) est disponible sur edunao et a été envoyé par mail.

Structure (conforme au §8.2 du sujet) :
1. Question de recherche et choix du dataset
2. Modèle de référence et symptôme observé
3. Hypothèse causale et expérience contrôlée
4. Correction proposée et évaluation
5. Menaces à la validité
6. Conclusion et apprentissages

---

##  Limitations / Limites identifiées

Conformément à l'exigence §7 du sujet, les principales menaces à la validité de nos conclusions sont :

1. **Non-significativité statistique** : avec n = 5 seeds, p = 0.13 ne permet pas de **prouver** l'équivalence F1(A) ≈ F1(E') — seulement de constater l'absence de différence détectable.
2. **Caractère borderline de l'overfitting initial** (gap = 0.128 modéré) : les conclusions ne se transposent pas mécaniquement à des configurations plus extrêmes.
3. **Confondants potentiels dans E'** : 5 hyperparamètres modifiés simultanément ; le modèle B (depth=2 seul) capture peut-être l'essentiel de l'effet.
4. **Pas de validation out-of-distribution** : un split temporel (train sur jan-sep, test sur oct-déc) renforcerait la démonstration.
5. **Choix de la métrique de gap** : `train_F1 − test_F1` ; d'autres définitions (log-loss, calibration) pourraient nuancer.

Voir §5 du rapport pour la discussion complète.

---

##  License & Acknowledgments

Project carried out as part of the *Introduction to AI and Machine Learning* course at École Centrale Casablanca (Spring 2026).

**Dataset citation :**
> Sakar, C.O., Polat, S.O., Katircioglu, M. *et al.* "Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks." *Neural Comput & Applic* 31, 6893–6908 (2019). [UCI ML Repository link](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

**Course material :** © UTER MID@S — École Centrale Casablanca.