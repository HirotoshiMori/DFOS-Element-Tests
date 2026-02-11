"""converter モジュールの単体テスト."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.converter import strain_to_shear, convert_columns_to_shear
from src.data_loader import POSITION_COLUMN


def test_strain_to_shear_zero() -> None:
    """ε=0 → γ=0."""
    assert strain_to_shear(0.0) == 0.0


def test_strain_to_shear_scalar() -> None:
    """スカラーで γ_s = sqrt((ε+1)² - 1)."""
    eps = 0.01
    expected = np.sqrt((eps + 1.0) ** 2 - 1.0)
    assert strain_to_shear(eps) == pytest.approx(expected)


def test_strain_to_shear_negative() -> None:
    """ε < 0 のとき γ_s = sqrt(1 - (ε+1)²)."""
    eps = -0.5
    expected = np.sqrt(1.0 - (eps + 1.0) ** 2)
    assert strain_to_shear(eps) == pytest.approx(expected)


def test_strain_to_shear_array() -> None:
    """配列入力で配列返却（正負混在）."""
    eps = np.array([0.0, 0.01, -0.5])
    out = strain_to_shear(eps)
    assert isinstance(out, np.ndarray)
    assert out[0] == 0.0
    assert out[1] == pytest.approx(np.sqrt((0.01 + 1.0) ** 2 - 1.0))
    assert out[2] == pytest.approx(np.sqrt(1.0 - (0.5) ** 2))  # ε=-0.5 → (ε+1)=0.5


def test_convert_columns_to_shear() -> None:
    """DataFrame のひずみ列がせん断ひずみに換算される."""
    df = pd.DataFrame({
        POSITION_COLUMN: [0.0, 0.05],
        "0mm": [0.0, 1e-5],
        "1mm": [1e-5, 2e-5],
    })
    out = convert_columns_to_shear(df, ["0mm", "1mm"])
    assert "shear_0mm" in out.columns
    assert "shear_1mm" in out.columns
    assert out["shear_0mm"].iloc[0] == 0.0
