"""Concrete per-source ingestors.

Each class subclasses :class:`ingestion.base.BaseIngestor` and implements
:meth:`fetch`. Import via :data:`SOURCE_REGISTRY` in the runner.
"""
from .kaggle_bankruptcy import KaggleBankruptcyIngestor
from .kaggle_news import KaggleNewsIngestor
from .kaggle_stock import KaggleStockIngestor
from .sec_edgar import SECEdgarIngestor
from .fred import FredIngestor
from .yahoo_finance import YahooFinanceIngestor
from .alpha_vantage import AlphaVantageIngestor
from .world_bank import WorldBankIngestor
from .imf import IMFIngestor
from .oecd import OECDIngestor

#: maps the config key under `sources:` to its ingestor class
SOURCE_REGISTRY = {
    "kaggle_bankruptcy": KaggleBankruptcyIngestor,
    "sec_edgar": SECEdgarIngestor,
    "fred": FredIngestor,
    "yahoo_finance": YahooFinanceIngestor,
    "alpha_vantage": AlphaVantageIngestor,
    "kaggle_news": KaggleNewsIngestor,
    "kaggle_stock": KaggleStockIngestor,
    "world_bank": WorldBankIngestor,
    "imf": IMFIngestor,
    "oecd": OECDIngestor,
}
