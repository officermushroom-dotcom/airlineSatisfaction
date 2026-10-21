# Kaggle Playground S6E10: Predicting Airline Satisfaction

My solution for Kaggle Playground Series Season 6 Episode 10.
The task is binary classification: predicting whether an airline passenger was satisfied, based on passenger, flight, and service rating data.

- Competition: https://www.kaggle.com/competitions/playground-series-s6e10
- Metric: ROC AUC
- Period: October 2026 (deadline: Oct 31)

## Data

- Train: ~700k rows / Test: ~300k rows
- Features: numerical features (age, flight distance, departure/arrival delays), 13 service ratings (ordinal, 0–5),
  and categorical features (gender, customer type, type of travel, class)
- Missing values only in `Arrival Delay in Minutes` (~0.04%)

*Data files are not included in this repository, in accordance with the competition rules.*

## Approach

- **Validation:** StratifiedKFold (5 folds, `random_state=42`). CV AUC is used as the primary metric for all decisions.
- **Model:** LightGBM with early stopping
- **Preprocessing:** Categorical columns are converted to a shared `category` dtype across train/test and handled natively by LightGBM.

## Experiment Log

| # | Description | CV AUC | Public LB |
|---|---|---|---|
| 1 | LightGBM baseline | 0.95886 | 0.95827 |
| 2 | + svc_mean, svc_min | 0.95863 | - |
| 3 | + original data | 0.95865 | - |
| 4 | CatBoost (GPU) baseline | 0.95846 | - |
| 5 | Blend: LightGBM + CatBoost | 0.95895 | 0.95836 |
| 6 |  LogisticRegression baseline | 0.94868 | - |
| 7 | blend: LightGBM + CatBoost + LogisticRegression | 0.95904 | 0.95847 |
| 8 | blend: tuned_LightGBM + CatBoost | 0.95916 | 0.95857 |
| 9 | tuned Catboost | 0.95852 | - |
| 10 | blend: tuned_LightGBM + tuned_CatBoost | 0.95910 | - |
| 11 | fd_encorded_LightGBM | 0.96054 | - |
| 12 | blend: fd_encoded_LightGBM + CatBoost | 0.96058 | 0.96017 |
| 13 | fd_encoded_catBoost | 0.96031 | - |
| 14 | blend: fd_encoded_LightGBM + fd_encoded_CatBoost | 0.96072 | 0.96033 |
| 15 | Stack 8 oof | 0.96071 | - |
| 16 | blend: seedavg_LightGBM_CatBoost | 0.96078 | 0.96035 |
| 17 | fd_encoded_XGBoost | 0.96031 | - |
| 18 | Stacking: own models + public TFM OOFs (Chris Deotte, 11 models), LR C=0.003 | 0.96149 | 0.96105 |

## How to Run

1. Download `train.csv`, `test.csv`, and `sample_submission.csv` from the competition page and place them in this directory.
2. Run `model.ipynb` from top to bottom.
3. Submit the generated `submission_*.csv`.

## Environment

- Python 3.14
- pandas, numpy, scikit-learn, lightgbm, matplotlib, seaborn

## Findings

### Validation
- **CV and Public LB stayed closely aligned throughout** (LB consistently ~0.0004–0.0006 below CV),
  so 5-fold stratified CV was used as the single source of truth for every decision.
  All models share the same split (`StratifiedKFold(5, shuffle=True, random_state=42)`),
  which makes their OOF predictions directly comparable and safe to blend or stack.

### What worked: Flight Distance as a route identifier (biggest gain)
- `Flight Distance` has only ~3.5k unique values across ~700k rows (~200 repeats per value).
  The values behave like **route IDs** rather than a continuous quantity: nearby distances do not
  share similar satisfaction rates.
- Adding a **frequency encoding** and an **in-fold target encoding** (smoothed with a = 20) improved
  LightGBM CV from **0.95903 → 0.96054 (+0.0015)**, with gains on every fold.
  This single change accounts for most of the total improvement in this project.
- Why it helps: a tree can only ask threshold questions (`x ≥ t`). A value-by-value pattern over
  thousands of distances would need hundreds of splits; the target encoding turns it into a single
  informative column.
- Leakage control: the target encoding is computed only from the training part of each fold,
  with an inner 5-fold split so that no row's own label enters its own encoding.
  The resulting CV–LB gap stayed unchanged, confirming there was no leakage.
- CatBoost benefits from the same idea when the distance is passed as a categorical feature
  (its ordered target statistics perform the encoding internally): **0.95846 → 0.96031**.

### What did not work (and why)
- **Service-rating aggregates** (mean/min, with or without treating 0 as "not applicable"),
  **encodings of other numeric columns** (Age, delays), and **Distance × Class / Travel / Customer
  combinations** all failed to improve CV.
  With ~700k rows, gradient-boosted trees already capture these relationships through ordinary
  splits; extra correlated columns only dilute feature sampling (`colsample_bytree`) and compete for splits.
- **Original (real) dataset**: neither appending its ~130k rows to training nor using a
  "teacher" model trained on it as a feature helped. Interestingly, the teacher alone reaches
  AUC 0.953 on the competition train set, so the real-data knowledge does transfer —
  the large synthetic training set simply already contains it.
- **Smoothing strength** of the target encoding was checked (a = 1–200); a = 20 was optimal.

### Models and ensembling
- **Hyperparameter tuning** (Optuna, 30 trials on one fold, verified on all 5 folds) improved
  LightGBM from 0.95886 to 0.95903, with gains on 4 of 5 folds.
  Tuning CatBoost improved it slightly alone but *lowered* the blend score, so it was not adopted —
  the tuned model became more similar to LightGBM, reducing ensemble diversity.
- **Blending LightGBM and CatBoost** consistently helped, because their tree-growing strategies
  (leaf-wise vs. symmetric trees) lead to different errors.
- **Logistic regression** (CV 0.94868) received a blend weight of 0: it added no information the
  tree models lacked.
- **Stacking our own 8 models** with a logistic-regression meta-model gave the same CV as a simple
  weighted average (0.96071 vs. 0.96072). The learned coefficients summed to ~1.0, i.e. the stack
  essentially rediscovered the weighted average — with only a few, highly similar models,
  stacking has nothing extra to exploit.
- **Seed averaging** (3 seeds each) gave small but reliable gains:
  LightGBM 0.96054 → 0.96064, CatBoost 0.96031 → 0.96043, blend → 0.96078.

### Stacking with public Tabular Foundation Model (TFM) predictions
- Using the publicly shared OOF/test predictions of 11 TFMs and baselines
  ([dataset by Chris Deotte](https://www.kaggle.com/datasets/cdeotte/s6e10-tfm-oof-and-test-predictions)), which were produced with the **same fold split** as ours,
  a logistic-regression stack was built in logit space (C = 0.003).
- | Stack | CV AUC |
  |---|---|
  | TFMs only | 0.96125 |
  | **Own models + TFMs** | **0.96149** (Public LB 0.96105) |
- Adding our models improved the TFM-only stack by **+0.00024**. The TFMs use only the 21 raw
  columns, so they lack the route-identifier information from the Flight Distance encoding —
  our models contribute exactly the signal the foundation models are missing.
- Note: this submission relies on publicly shared predictions. Our self-built pipeline alone
  reaches CV 0.96078 / LB 0.9603x; the TFM stack is reported separately to make our own
  contribution explicit.