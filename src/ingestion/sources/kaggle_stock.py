"""Kaggle Stock Market (OHLCV) dataset — time-series for forecasting."""
from __future__ import annotations

import os

import pandas as pd

from ..base import BaseIngestor, IngestionError
from ._kaggle_helper import download_kaggle

OHLCV = ["Date", "Open", "High", "Low", "Close", "Volume"]


class KaggleStockIngestor(BaseIngestor):
    name = "kaggle_stock"

    def fetch(self) -> pd.DataFrame:
        csvs = download_kaggle(self.config["dataset"], self._raw_dir(), self.log)
        # This dataset can contain thousands of per-ticker files. To stay bounded,
        # read up to `max_files` and tag each row with its ticker (the filename).
        max_files = int(self.config.get("max_files", 50))
        frames = []
        for path in csvs[:max_files]:
            try:
                part = pd.read_csv(path)
            except Exception as exc:
                self.log.warning("skip %s (%s)", path, exc)
                continue
            ticker = os.path.splitext(os.path.basename(path))[0].split(".")[0]
            part["ticker"] = ticker.upper()
            frames.append(part)
        if not frames:
            raise IngestionError("no readable OHLCV files")
        if len(csvs) > max_files:
            self.log.warning("capped at %d of %d files (config.max_files)",
                             max_files, len(csvs))
        df = pd.concat(frames, ignore_index=True)
        df.columns = [c.strip() for c in df.columns]
        return df
