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

## How to Run

1. Download `train.csv`, `test.csv`, and `sample_submission.csv` from the competition page and place them in this directory.
2. Run `model.ipynb` from top to bottom.
3. Submit the generated `submission_*.csv`.

## Environment

- Python 3.14
- pandas, numpy, scikit-learn, lightgbm, matplotlib, seaborn

## Findings

- **Target encoding works only for Flight Distance (Exp 9–12):** Flight Distance has ~3.5k unique values
  that repeat ~200 times each, behaving like a route identifier rather than a continuous number.
  Frequency + in-fold target encoding improved LightGBM CV from 0.95903 to 0.96054.
  Extending the same encoding to Age and delay columns, or to combinations such as Distance × Class,
  did not help — tree models already capture those relationships through ordinary splits.
  The smoothing strength (a = 20) was confirmed to be optimal.