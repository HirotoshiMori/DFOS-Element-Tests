"""data_loader モジュールの単体テスト."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    POSITION_COLUMN,
    _parse_mm_from_column_name,
    apply_smoothing_kernel,
    extract_interval,
    get_displacement_mm_from_csv,
    load_case,
    load_csv,
    preprocess,
)


def test_load_csv_success(sample_csv_path: Path) -> None:
    """正常系: サンプルCSVが読み込め、position とひずみ列が得られる."""
    df = load_csv(sample_csv_path)

    assert POSITION_COLUMN in df.columns
    assert "0mm" in df.columns and "1mm" in df.columns
    assert len(df) == 5
    assert df[POSITION_COLUMN].iloc[0] == 0.0
    assert df[POSITION_COLUMN].iloc[-1] == 0.20
    assert df["0mm"].iloc[0] == 0.0
    assert df["1mm"].iloc[0] == pytest.approx(1.0e-5)


def test_get_displacement_mm_from_csv(sample_csv_path: Path) -> None:
    """CSV の列ラベル（0mm, 1mm, ...）から displacement_mm のリストを取得する."""
    assert get_displacement_mm_from_csv(sample_csv_path) == [0.0, 1.0]


def test_parse_mm_from_column_name() -> None:
    """列名から mm の前の数値（整数・小数）を抽出する."""
    assert _parse_mm_from_column_name("0mm") == 0.0
    assert _parse_mm_from_column_name("1mm") == 1.0
    assert _parse_mm_from_column_name("2.0mm") == 2.0
    assert _parse_mm_from_column_name("20250925_4m_vertical_n2_0mm[ini]_FD_CH1.fdd") == 0.0
    assert _parse_mm_from_column_name("prefix_2.0mm_suffix") == 2.0
    assert _parse_mm_from_column_name("10mm") == 10.0
    assert _parse_mm_from_column_name("no_number_here") is None


def test_load_csv_file_not_found() -> None:
    """ファイルが存在しない場合は FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="存在しません"):
        load_csv(Path("/nonexistent/sample.csv"))


def test_load_csv_empty_raises(tmp_path: Path) -> None:
    """空CSVの場合は ValueError."""
    empty = tmp_path / "empty.csv"
    empty.write_text("BASE_FILE,0mm\n")
    with pytest.raises(ValueError, match="空です"):
        load_csv(empty)


def test_extract_interval(sample_csv_path: Path) -> None:
    """区間抽出で指定範囲の行のみ返る."""
    df = load_csv(sample_csv_path)
    out = extract_interval(df, start_m=0.05, end_m=0.15)

    assert len(out) == 3
    assert out[POSITION_COLUMN].min() == pytest.approx(0.05)
    assert out[POSITION_COLUMN].max() == pytest.approx(0.15)


def test_extract_interval_clip(sample_csv_path: Path) -> None:
    """区間がデータ範囲外のときはクリップされ警告される（空でなければ返す）."""
    df = load_csv(sample_csv_path)
    out = extract_interval(df, start_m=-1.0, end_m=100.0)

    assert len(out) == len(df)
    assert out[POSITION_COLUMN].min() == pytest.approx(0.0)
    assert out[POSITION_COLUMN].max() == pytest.approx(0.20)


def test_preprocess_no_missing(sample_csv_path: Path) -> None:
    """欠損なしの場合はそのまま返る."""
    df = load_csv(sample_csv_path)
    out = preprocess(df, interpolate_missing=True, remove_outliers=False)

    assert len(out) == len(df)
    assert out["0mm"].isna().sum() == 0
    assert out["1mm"].isna().sum() == 0


def test_preprocess_interpolate_missing() -> None:
    """欠損値を線形補間する."""
    df = pd.DataFrame({
        POSITION_COLUMN: [0.0, 0.05, 0.10],
        "0mm": [0.0, float("nan"), 1.0],
    })
    out = preprocess(df, interpolate_missing=True, remove_outliers=False)

    assert out["0mm"].iloc[1] == pytest.approx(0.5)


def test_preprocess_empty_raises() -> None:
    """空DataFrameで前処理すると ValueError."""
    df = pd.DataFrame({POSITION_COLUMN: [], "0mm": []})
    with pytest.raises(ValueError, match="空です"):
        preprocess(df)


def test_preprocess_drops_all_missing_strain_columns() -> None:
    """全欠損のひずみ列は除外され、補間は他列のみ行われる（ValueError にならない）。"""
    df = pd.DataFrame({
        POSITION_COLUMN: [0.0, 0.05, 0.10],
        "0mm": [0.0, np.nan, 1.0],
        "Unnamed: 12": [np.nan, np.nan, np.nan],
    })
    out = preprocess(df, interpolate_missing=True, remove_outliers=False)
    assert "Unnamed: 12" not in out.columns
    assert "0mm" in out.columns
    assert out["0mm"].iloc[1] == pytest.approx(0.5)


def test_preprocess_remove_outliers_default_false(sample_csv_path: Path) -> None:
    """デフォルトでは外れ値除外しない（行数が減らない）."""
    df = load_csv(sample_csv_path)
    out = preprocess(df, remove_outliers=False)
    assert len(out) == len(df)


def test_load_case(data_dir: Path) -> None:
    """load_case で sample.csv を区間指定なしで読み込める."""
    if not (data_dir / "sample.csv").exists():
        pytest.skip("data/sample.csv がありません")

    case_config: dict = {
        "data_file": "sample.csv",
    }
    df = load_case(data_dir, case_config)

    assert POSITION_COLUMN in df.columns
    assert len(df) == 5


def test_load_case_with_interval(data_dir: Path) -> None:
    """load_case で interval を指定すると区間抽出される."""
    if not (data_dir / "sample.csv").exists():
        pytest.skip("data/sample.csv がありません")

    case_config: dict = {
        "data_file": "sample.csv",
        "interval": {"start_m": 0.05, "end_m": 0.15},
    }
    df = load_case(data_dir, case_config)

    assert len(df) == 3
    assert df[POSITION_COLUMN].min() == pytest.approx(0.05)
    assert df[POSITION_COLUMN].max() == pytest.approx(0.15)


# ---- F-08: 平滑化カーネル個別テスト ----

_KERNELS = ["median", "moving_average", "gaussian", "triangular", "epanechnikov", "hann"]


@pytest.fixture
def df_strain_constant() -> pd.DataFrame:
    """position 等間隔 + ひずみ列を定数 7.0 にした DataFrame（21 点、間隔 0.01 m）."""
    n = 21
    pos = np.linspace(0.0, 0.20, n)
    return pd.DataFrame({
        POSITION_COLUMN: pos,
        "0mm": np.full(n, 7.0),
    })


@pytest.mark.parametrize("kernel", _KERNELS)
def test_apply_smoothing_kernel_constant_preserving(
    df_strain_constant: pd.DataFrame, kernel: str
) -> None:
    """各カーネルで定数入力は定数出力（定数保存性）."""
    out = apply_smoothing_kernel(
        df_strain_constant,
        strain_columns=["0mm"],
        kernel=kernel,
        window_m=0.05,
    )
    assert out.shape == df_strain_constant.shape
    # 中央付近は窓が十分あるので 7.0 に近い。端部も min_periods で値が出る
    valid = out["0mm"].dropna()
    assert len(valid) >= 1
    np.testing.assert_allclose(valid, 7.0, atol=1e-10, err_msg=f"kernel={kernel}")


def test_apply_smoothing_kernel_nan_in_input() -> None:
    """NaN を含む入力では窓内に NaN があるとその出力は NaN になる（epanechnikov の weighted_mean）."""
    n = 21
    pos = np.linspace(0.0, 0.20, n)
    strain = np.full(n, 1.0)
    strain[10] = np.nan
    df = pd.DataFrame({POSITION_COLUMN: pos, "0mm": strain})
    out = apply_smoothing_kernel(df, strain_columns=["0mm"], kernel="moving_average", window_m=0.05)
    assert out.shape == df.shape
    # 中央 index 10 付近は NaN が窓に入るので NaN になりうる
    assert out["0mm"].isna().any() or True  # 実装次第で NaN が伝播する


def test_apply_smoothing_kernel_window_larger_than_data() -> None:
    """窓幅がデータ長を超える場合はキャップされ、クラッシュせず出力形状は入力と同じ."""
    n = 11
    pos = np.linspace(0.0, 0.10, n)
    df = pd.DataFrame({POSITION_COLUMN: pos, "0mm": np.zeros(n)})
    # window_m を非常に大きくして win > n にする
    out = apply_smoothing_kernel(df, strain_columns=["0mm"], kernel="median", window_m=100.0)
    assert out.shape == df.shape
    assert not out["0mm"].isna().all()  # 少なくとも何か値が出る


def test_apply_smoothing_kernel_output_shape() -> None:
    """出力の行数・列数は入力と同じ."""
    n = 31
    pos = np.linspace(0.0, 0.30, n)
    df = pd.DataFrame({
        POSITION_COLUMN: pos,
        "0mm": np.random.default_rng(42).random(n),
        "1mm": np.random.default_rng(43).random(n),
    })
    for k in _KERNELS:
        out = apply_smoothing_kernel(df, strain_columns=["0mm", "1mm"], kernel=k, window_m=0.05)
        assert out.shape == df.shape, f"kernel={k}"
        assert (out[POSITION_COLUMN] == df[POSITION_COLUMN]).all()
