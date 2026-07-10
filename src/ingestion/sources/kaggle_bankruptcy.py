"""Kaggle Company Bankruptcy Prediction — the primary LABELED dataset."""
from __future__ import annotations

import pandas as pd

from ..base import BaseIngestor
from ._kaggle_helper import download_kaggle


class KaggleBankruptcyIngestor(BaseIngestor):
    name = "kaggle_bankruptcy"

    def fetch(self) -> pd.DataFrame:
        csvs = download_kaggle(self.config["dataset"], self._raw_dir(), self.log)
        # this dataset ships a single CSV
        df = pd.read_csv(csvs[0])
        df.columns = [c.strip() for c in df.columns]  # strip leading spaces
        return df

    def required_columns(self):
        return [self.config.get("target_col", "Bankrupt?")]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates().reset_index(drop=True)
        target = self.config.get("target_col", "Bankrupt?")
        if target in df.columns:
            df[target] = df[target].astype(int)
        return df
