"""Stage 2 — Data Collection.

Auto-downloads the Kaggle 'Company Bankruptcy Prediction' dataset via the
Kaggle API, unzips it, and returns the raw DataFrame.

Setup (one time):
    pip install kaggle
    # Kaggle -> Account -> Create New API Token  => downloads kaggle.json
    mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
"""
from __future__ import annotations

import glob
import os
import zipfile

import pandas as pd


def download_kaggle_dataset(slug: str, raw_dir: str) -> str:
    """Download + unzip a Kaggle dataset. Returns path to the extracted CSV."""
    os.makedirs(raw_dir, exist_ok=True)

    # Skip download if a CSV is already present.
    existing = glob.glob(os.path.join(raw_dir, "*.csv"))
    if existing:
        print(f"[data] Using cached CSV: {existing[0]}")
        return existing[0]

    # Import here so the rest of the pipeline works even without kaggle installed.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json
    print(f"[data] Downloading '{slug}' from Kaggle ...")
    api.dataset_download_files(slug, path=raw_dir, quiet=False)

    # dataset_download_files writes <name>.zip — extract it.
    for zf in glob.glob(os.path.join(raw_dir, "*.zip")):
        with zipfile.ZipFile(zf) as z:
            z.extractall(raw_dir)
        os.remove(zf)

    csvs = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {raw_dir} after download.")
    print(f"[data] Saved: {csvs[0]}")
    return csvs[0]


def load(cfg: dict) -> pd.DataFrame:
    """Entry point used by the runner."""
    csv_path = download_kaggle_dataset(
        cfg["data"]["kaggle_dataset"], cfg["data"]["raw_dir"]
    )
    df = pd.read_csv(csv_path)
    # Kaggle CSV columns often carry a leading space — normalize whitespace.
    df.columns = [c.strip() for c in df.columns]
    print(f"[data] Loaded {df.shape[0]} rows x {df.shape[1]} cols")
    return df
