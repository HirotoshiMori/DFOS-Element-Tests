"""伸縮ひずみ ε → せん断ひずみ γ の幾何換算. Python 3.12."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import POSITION_COLUMN


def strain_to_shear(
    epsilon: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    伸縮ひずみ ε をせん断ひずみ γ_s に換算する.
    桁落ち回避のため ε(ε+2) 形式を使用.
    ε >= 0: γ_s = √(ε(ε+2)).
    ε < 0:  γ_s = √(-ε(ε+2)).

    Args:
        epsilon: 伸縮ひずみ（無次元）.

    Returns:
        せん断ひずみ（無次元）. 入力の型に合わせて返す.
    """
    if isinstance(epsilon, pd.Series):
        return pd.Series(strain_to_shear(epsilon.values), index=epsilon.index)
    arr = np.asarray(epsilon, dtype=float)
    out = np.empty_like(arr)
    mask_nonneg = arr >= 0
    # ε(ε+2) = (ε+1)²−1 と等価。桁落ち回避。
    out[mask_nonneg] = np.sqrt(arr[mask_nonneg] * (arr[mask_nonneg] + 2.0))
    out[~mask_nonneg] = np.sqrt(-arr[~mask_nonneg] * (arr[~mask_nonneg] + 2.0))
    if np.isscalar(epsilon):
        return float(out)
    return out


def convert_columns_to_shear(
    df: pd.DataFrame,
    strain_columns: list[str],
) -> pd.DataFrame:
    """
    指定したひずみ列をせん断ひずみに換算し、shear_<列名> として返す.

    Args:
        df: position 列 + ひずみ列の DataFrame.
        strain_columns: 換算対象の列名リスト.

    Returns:
        position 列 + shear_<列名> の DataFrame.
    """
    out = df[[POSITION_COLUMN]].copy() if POSITION_COLUMN in df.columns else pd.DataFrame()
    for col in strain_columns:
        if col not in df.columns:
            continue
        out[f"shear_{col}"] = strain_to_shear(df[col])
    return out
