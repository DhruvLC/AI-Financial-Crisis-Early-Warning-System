"""Shared helper for Kaggle-dataset downloads (used by 3 ingestors)."""
from __future__ import annotations

import glob
import os
import zipfile

from ..base import IngestionError


def download_kaggle(slug: str, dest_dir: str, logger) -> list[str]:
    """Download + unzip a Kaggle dataset. Returns list of extracted CSV paths.

    Skips the API call if CSVs are already present (offline / cached use).
    Raises IngestionError with an actionable message if credentials are absent.
    """
    os.makedirs(dest_dir, exist_ok=True)
    cached = glob.glob(os.path.join(dest_dir, "*.csv"))
    if cached:
        logger.info("using cached CSVs in %s", dest_dir)
        return cached

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # library missing
        raise IngestionError(f"kaggle library not installed: {exc}") from exc

    try:
        api = KaggleApi()
        api.authenticate()  # reads ~/.kaggle/kaggle.json
        logger.info("downloading kaggle dataset '%s'", slug)
        api.dataset_download_files(slug, path=dest_dir, quiet=True)
    except Exception as exc:
        raise IngestionError(
            f"kaggle download failed for '{slug}'. Ensure ~/.kaggle/kaggle.json "
            f"exists (chmod 600). Original error: {exc}"
        ) from exc

    for zf in glob.glob(os.path.join(dest_dir, "*.zip")):
        with zipfile.ZipFile(zf) as z:
            z.extractall(dest_dir)
        os.remove(zf)

    csvs = glob.glob(os.path.join(dest_dir, "**", "*.csv"), recursive=True)
    if not csvs:
        raise IngestionError(f"no CSV found after downloading '{slug}'")
    return csvs
