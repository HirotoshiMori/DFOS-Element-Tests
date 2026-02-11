"""理論値との誤差評価（RMSE, 最大誤差, 平均誤差, 相関係数）. Python 3.12."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE: √(mean((y_true - y_pred)²))."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """最大絶対誤差: max(|y_true - y_pred|)."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.max(np.abs(diff)))


def mean_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均誤差: mean(y_pred - y_true)."""
    return float(np.mean(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """相関係数. 長さ1のときは 1.0、分散0（定数）のときは 0.0 を返す."""
    t = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    if t.size != p.size or t.size < 2:
        return 1.0 if t.size == p.size and t.size == 1 else 0.0
    if np.std(t) == 0 or np.std(p) == 0:
        return 0.0
    c = np.corrcoef(t, p)[0, 1]
    return float(c) if not np.isnan(c) else 0.0


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    上記4指標を一括計算する.

    Returns:
        {"rmse", "max_error", "mean_error", "correlation"}
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
        "mean_error": mean_error(y_true, y_pred),
        "correlation": correlation(y_true, y_pred),
    }


def build_metrics_table(results: list[dict[str, float]]) -> pd.DataFrame:
    """
    各手法の評価結果のリストから、手法名・RMSE・最大誤差・平均誤差・相関係数の DataFrame を生成する.

    Args:
        results: [{"method": str, "rmse": float, ...}, ...]

    Returns:
        指標の列を持つ DataFrame.
    """
    if not results:
        return pd.DataFrame(columns=["method", "rmse", "max_error", "mean_error", "correlation"])
    return pd.DataFrame(results)
