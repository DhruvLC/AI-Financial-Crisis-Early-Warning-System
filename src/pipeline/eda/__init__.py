"""Exploratory Data Analysis (EDA) module.

A modular, config-driven EDA framework mirroring the design of
``src/pipeline/preprocessing`` and ``src/validation``: one
:class:`~.base.EdaAnalyzer` subclass per concern (overview, target, descriptive
stats, missing values, outliers, distributions, correlation, financial ratios,
feature–target relationships, dimensionality), a :class:`~.runner.EdaRunner`
that sequences them, a :class:`~.insights.BusinessInsightsEngine` that distills
findings, publication-quality figures via :class:`~.plotting.FigureManager`, and
a multi-format (JSON/Markdown/HTML/CSV) :class:`~.report.EdaReport` writer.
"""
from .base import AnalysisResult, EdaAnalyzer, EdaError
from .insights import BusinessInsightsEngine
from .plotting import FigureManager
from .report import EdaReport
from .runner import EdaResult, EdaRunner
from .analyzers import ANALYZER_REGISTRY, DEFAULT_ORDER

__all__ = [
    "AnalysisResult", "EdaAnalyzer", "EdaError",
    "BusinessInsightsEngine", "FigureManager", "EdaReport",
    "EdaResult", "EdaRunner",
    "ANALYZER_REGISTRY", "DEFAULT_ORDER",
]
