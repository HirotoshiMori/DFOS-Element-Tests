"""evaluator モジュールの単体テスト."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluator import (
    rmse,
    max_error,
    mean_error,
    correlation,
    evaluate_all,
    build_metrics_table,
)


def test_rmse_identical() -> None:
    """y_true == y_pred のとき RMSE=0."""
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_max_error() -> None:
    """最大絶対誤差が正しく計算される."""
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 2.5])
    assert max_error(y_true, y_pred) == 0.5


def test_correlation_perfect() -> None:
    """完全一致で相関=1."""
    y = np.array([1.0, 2.0, 3.0])
    assert correlation(y, y) == 1.0


def test_evaluate_all() -> None:
    """evaluate_all が4キーを返す."""
    y = np.array([0.0, 1.0, 2.0])
    out = evaluate_all(y, y)
    assert out["rmse"] == 0.0
    assert out["correlation"] == 1.0
    assert "max_error" in out and "mean_error" in out


def test_build_metrics_table() -> None:
    """build_metrics_table が DataFrame を返す."""
    results = [
        {"method": "a", "rmse": 0.01, "max_error": 0.02, "mean_error": 0.0, "correlation": 0.99},
    ]
    df = build_metrics_table(results)
    assert len(df) == 1
    assert df["rmse"].iloc[0] == 0.01
