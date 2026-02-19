# Download Titanic dataset into data/ using kagglehub or kaggle API.
# Run: python scripts/download_data.py

from pathlib import Path
import os
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "yasserh/titanic-dataset"
TARGET = DATA_DIR / "Titanic-Dataset.csv"

def download_with_kagglehub() -> bool:
    try:
        import kagglehub
    except ModuleNotFoundError:
        return False

    path = kagglehub.dataset_download(DATASET)
    src = Path(path) / "Titanic-Dataset.csv"
    if not src.exists():
        raise RuntimeError(f"Expected file not found in kagglehub path: {src}")
    TARGET.write_bytes(src.read_bytes())
    print(f"Downloaded dataset via kagglehub to {TARGET}")
    return True


def download_with_kaggle_api() -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError:
        return False

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET, path=str(DATA_DIR), unzip=True)
    if not TARGET.exists():
        raise RuntimeError(f"Download finished but file not found at {TARGET}")
    print(f"Downloaded dataset via kaggle API to {TARGET}")
    return True


if TARGET.exists():
    print(f"Dataset already exists at {TARGET}")
    sys.exit(0)

if download_with_kagglehub():
    sys.exit(0)

if download_with_kaggle_api():
    sys.exit(0)

raise RuntimeError(
    "Unable to download dataset. Install `kagglehub` or `kaggle`, "
    "and ensure Kaggle credentials are configured (KAGGLE_USERNAME/KAGGLE_KEY)."
)
