"""解析パイプライン用カスタム例外. Python 3.12."""

from __future__ import annotations


class AnalysisError(Exception):
    """解析エラーの基底クラス."""


class DataLoadError(AnalysisError):
    """データ読み込みエラー."""


class ValidationError(AnalysisError):
    """設定検証エラー."""


class CalculationError(AnalysisError):
    """計算エラー."""
