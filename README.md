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

## How to Run

1. Download `train.csv`, `test.csv`, and `sample_submission.csv` from the competition page and place them in this directory.
2. Run `model.ipynb` from top to bottom.
3. Submit the generated `submission_*.csv`.

## Environment

- Python 3.14
- pandas, numpy, scikit-learn, lightgbm, matplotlib, seaborn