"""reporter モジュールの単体テスト."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.reporter import (
    aggregate_results,
    create_summary_csv,
    plot_comparison,
    plot_multi_case_results,
)


def test_aggregate_results_empty(tmp_path: Path) -> None:
    """出力ディレクトリが空なら空 DataFrame."""
    df = aggregate_results(tmp_path)
    assert df.empty


def test_aggregate_results_from_summary(tmp_path: Path) -> None:
    """summary.csv があればそれを読む."""
    summary = tmp_path / "summary.csv"
    summary.write_text("case_id,rmse,max_error\nc1,0.01,0.02\n")
    df = aggregate_results(tmp_path)
    assert len(df) == 1
    assert df["case_id"].iloc[0] == "c1"
    assert df["rmse"].iloc[0] == pytest.approx(0.01)


def test_create_summary_csv(tmp_path: Path) -> None:
    """create_summary_csv でファイルができる."""
    results = pd.DataFrame([{"case_id": "a", "rmse": 0.1}])
    out = tmp_path / "out" / "summary.csv"
    create_summary_csv(results, out)
    assert out.exists()
    assert "case_id" in out.read_text()


def test_plot_comparison(tmp_path: Path) -> None:
    """plot_comparison で PNG が作成される."""
    results = pd.DataFrame({"case_id": ["c1", "c2"], "rmse": [0.01, 0.02]})
    out = tmp_path / "comparison.png"
    plot_comparison(results, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_multi_case_results(tmp_path: Path) -> None:
    """plot_multi_case_results で複数サブプロットの PNG が作成される."""
    cases_data = [
        {
            "case_id": "case_01",
            "displacement_mm": [0, 1, 2],
            "theory_shear_strain": [0.0, 0.001, 0.002],
            "predicted_by_kernel": {"point": [0.0, 0.001, 0.002], "median": [0.0, 0.0009, 0.0021]},
        },
        {
            "case_id": "case_02",
            "displacement_mm": [0, 1],
            "theory_shear_strain": [0.0, 0.001],
            "predicted_by_kernel": {"point": [0.0, 0.001]},
        },
    ]
    out = tmp_path / "result.png"
    plot_multi_case_results(cases_data, out)
    assert out.exists()
    assert out.stat().st_size > 0
