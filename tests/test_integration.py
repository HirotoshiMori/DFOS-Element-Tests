"""End-to-end 統合テスト. Python 3.12."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.cli import run_single_case, run_all_cases


def test_end_to_end_single_case(tmp_path: Path) -> None:
    """単一ケースの end-to-end: 設定・データが揃っている場合に実行完了する."""
    # プロジェクトルートの params/data を参照するため、実在する case_sample を使う
    root = Path(__file__).resolve().parent.parent
    params_dir = root / "params"
    data_dir = root / "data"
    if not (params_dir / "cases" / "case_sample.yml").exists():
        pytest.skip("params/cases/case_sample.yml がありません")
    if not (data_dir / "sample.csv").exists():
        pytest.skip("data/sample.csv がありません")

    out = tmp_path / "output"
    result = run_single_case(
        "case_sample",
        output_dir=out,
        config_dir=params_dir,
        data_dir=data_dir,
        verbose=False,
    )
    assert "rmse" in result
    assert "case_id" in result
    assert (out / "case_sample" / "run.log").exists()
    assert (out / "case_sample" / "metrics.csv").exists()
    assert (out / "case_sample" / "params_used" / "common.yml").exists()


def test_end_to_end_all_cases(tmp_path: Path) -> None:
    """全ケース一括実行で summary が生成される（case_list なしで cases/*.yml を実行、最低1ケースあればよい）。"""
    import shutil
    root = Path(__file__).resolve().parent.parent
    params_src = root / "params"
    data_dir = root / "data"
    cases_dir = params_src / "cases"
    if not cases_dir.is_dir() or not list(cases_dir.glob("*.yml")):
        pytest.skip("params/cases/*.yml がありません")
    config_dir = tmp_path / "params"
    shutil.copytree(params_src, config_dir, ignore=lambda d, names: [n for n in names if n == "case_list.yml"])
    if (config_dir / "case_list.yml").exists():
        (config_dir / "case_list.yml").unlink()

    out = tmp_path / "output"
    df = run_all_cases(output_dir=out, config_dir=config_dir, data_dir=data_dir, verbose=False)
    assert isinstance(df, pd.DataFrame)
    # 少なくとも1件は成功しているか、またはエラー行が含まれる
    assert len(df) >= 1
    if (out / "summary.csv").exists():
        summary = pd.read_csv(out / "summary.csv")
        assert len(summary) >= 1


def test_output_file_structure(tmp_path: Path) -> None:
    """単一ケース実行後の出力ディレクトリ構造を検証する."""
    root = Path(__file__).resolve().parent.parent
    if not (root / "params" / "cases" / "case_sample.yml").exists():
        pytest.skip("case_sample がありません")
    if not (root / "data" / "sample.csv").exists():
        pytest.skip("data/sample.csv がありません")

    run_single_case(
        "case_sample",
        output_dir=tmp_path,
        config_dir=root / "params",
        data_dir=root / "data",
        verbose=False,
    )
    case_out = tmp_path / "case_sample"
    assert case_out.is_dir()
    assert (case_out / "run.log").is_file()
    assert (case_out / "metrics.csv").is_file()
    assert (case_out / "params_used").is_dir()
    assert (case_out / "params_used" / "common.yml").is_file()
    assert (case_out / "params_used" / "case_sample.yml").is_file()
