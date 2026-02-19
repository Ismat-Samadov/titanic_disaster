# Usage Guide

This project expects the Titanic dataset CSV to be available locally.

## Prerequisites

- Python 3.8+
- Dependencies installed from `scripts/requirements.txt`

## Data Setup

Place the dataset at:

`data/Titanic-Dataset.csv`

## Run

```bash
python scripts/generate_charts.py
```

## Outputs

- Charts are written to `charts/` as PNG files.
- Console output includes EDA summaries and model metrics.

## Notes

- The script is configured to use a non-interactive Matplotlib backend and will not open GUI windows.
- Hyperparameter tuning runs in a single process to avoid joblib permission issues in restricted environments.
