# Technical Summary

## Pipeline Overview

1. Load `data/Titanic-Dataset.csv`.
2. Clean and impute missing values.
3. Engineer features (title extraction, family size, age groups, fare bins).
4. Encode categorical variables.
5. Train/test split with stratification and standard scaling.
6. Train multiple models and compare metrics.
7. Hyperparameter tune top performers via `RandomizedSearchCV`.
8. Generate evaluation charts and summary metrics.

## Data Cleaning

- `Age`: median imputation by `Sex` and `Pclass`.
- `Embarked`: mode imputation.
- `Fare`: median imputation.
- `Cabin`: converted to binary `HasCabin`.

## Feature Engineering

- `Title` extracted from `Name` and mapped into common/rare categories.
- `FamilySize = SibSp + Parch + 1`.
- `IsAlone = FamilySize == 1`.
- `AgeGroup` binned into: Child, Teen, Adult, Middle, Senior.
- `FareBin` quartile bins.

## Encoding

- `Sex` mapped to numeric (male=0, female=1).
- One-hot encoding for `Embarked` and `Title` (drop first).

## Models

Baseline models:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM
- KNN
- Naive Bayes
- AdaBoost
- XGBoost (optional, if installed)
- LightGBM (optional, if installed)

Hyperparameter tuning:
- `RandomizedSearchCV`, 20 iterations, 5-fold CV.
- `n_jobs=1` to avoid joblib multiprocessing constraints.

## Outputs

- EDA charts (missing values, survival by features).
- Model comparison chart.
- Confusion matrix and ROC curve for best baseline model.
- Baseline vs optimized model comparison.
- Final confusion matrix and ROC curve for best optimized model.

## Files

- Script: `scripts/generate_charts.py`
- Output charts: `charts/*.png`
