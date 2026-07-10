"""Kaggle Financial News dataset — headlines for future NLP sentiment."""
from __future__ import annotations

import pandas as pd

from ..base import BaseIngestor
from ._kaggle_helper import download_kaggle


class KaggleNewsIngestor(BaseIngestor):
    name = "kaggle_news"

    def fetch(self) -> pd.DataFrame:
        csvs = download_kaggle(self.config["dataset"], self._raw_dir(), self.log)
        # concatenate all CSVs the dataset ships; keep source filename as a column
        frames = []
        for path in csvs:
            part = pd.read_csv(path, on_bad_lines="skip")
            part["_source_file"] = path.rsplit("/", 1)[-1]
            frames.append(part)
        df = pd.concat(frames, ignore_index=True)
        df.columns = [c.strip() for c in df.columns]
        return df
