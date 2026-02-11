"""CLI の単体テスト. Python 3.12."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from src.cli import create_parser, cmd_compare, cmd_validate, run_compare, run_single_case
from src.exceptions import ValidationError


def test_create_parser() -> None:
    """パーサーに run / validate / compare がある."""
    parser = create_parser()
    args = parser.parse_args(["run", "--case", "case_01"])
    assert args.command == "run"
    assert args.case == ["case_01"]
    args2 = parser.parse_args(["validate", "--config", "params/common.yml"])
    assert args2.command == "validate"
    args3 = parser.parse_args(["compare", "--cases", "case_01", "case_02"])
    assert args3.command == "compare"
    assert args3.cases == ["case_01", "case_02"]


def test_cmd_validate_ok(tmp_path: Path) -> None:
    """正常な YAML で validate が 0 を返す."""
    common = tmp_path / "common.yml"
    common.write_text("smoothing: { window_m: 0.1 }\nplot: {}\n")
    args = Namespace(config=common, case=None, verbose=False)
    assert cmd_validate(args) == 0


def test_cmd_validate_missing_config() -> None:
    """存在しない設定で 1 を返す."""
    args = Namespace(config=Path("/nonexistent/common.yml"), case=None, verbose=False)
    assert cmd_validate(args) == 1


def test_cli_validate_config(tmp_path: Path) -> None:
    """CLI 設定検証: 正常 YAML で 0 を返す."""
    common = tmp_path / "common.yml"
    common.write_text("smoothing: { window_m: 0.1 }\nplot: {}\n")
    args = Namespace(config=common, case=None, verbose=False)
    assert cmd_validate(args) == 0


def test_cli_run_single_case(tmp_path: Path) -> None:
    """CLI 単一ケース実行: case_sample が存在すれば実行して結果を返す."""
    root = Path(__file__).resolve().parent.parent
    if not (root / "params" / "cases" / "case_sample.yml").exists():
        pytest.skip("case_sample がありません")
    if not (root / "data" / "sample.csv").exists():
        pytest.skip("data/sample.csv がありません")
    result = run_single_case(
        "case_sample",
        output_dir=tmp_path / "out",
        config_dir=root / "params",
        data_dir=root / "data",
        verbose=False,
    )
    assert result and isinstance(result, list) and len(result) > 0 and "rmse" in result[0]
    assert (tmp_path / "out" / "case_sample" / "metrics.csv").exists()


def test_cli_run_all_cases(tmp_path: Path) -> None:
    """CLI 全ケース実行: case_list なしで cases/*.yml を実行し、少なくとも1件あれば DataFrame が返る."""
    import shutil
    root = Path(__file__).resolve().parent.parent
    params_src = root / "params"
    cases_dir = params_src / "cases"
    if not cases_dir.is_dir() or not list(cases_dir.glob("*.yml")):
        pytest.skip("params/cases/*.yml がありません")
    # case_list.yml があるとそちらが優先されるため、case_list なしの設定ディレクトリを用意
    config_dir = tmp_path / "params"
    shutil.copytree(params_src, config_dir, ignore=lambda d, names: [n for n in names if n == "case_list.yml"])
    if (config_dir / "case_list.yml").exists():
        (config_dir / "case_list.yml").unlink()
    from src.cli import run_all_cases
    df = run_all_cases(
        output_dir=tmp_path / "out",
        config_dir=config_dir,
        data_dir=root / "data",
        verbose=False,
    )
    assert len(df) >= 1
    assert "rmse" in df.columns or "case_id" in df.columns


def test_cli_invalid_case(tmp_path: Path) -> None:
    """存在しないケース指定で ValidationError."""
    root = Path(__file__).resolve().parent.parent
    common = root / "params" / "common.yml"
    if not common.exists():
        pytest.skip("params/common.yml がありません")
    case_path = root / "params" / "cases" / "nonexistent_case_xyz.yml"
    if case_path.exists():
        pytest.skip("テスト対象のケースが誤って存在する")
    with pytest.raises(ValidationError, match="存在しません|ケース設定"):
        run_single_case(
            "nonexistent_case_xyz",
            output_dir=tmp_path,
            config_dir=root / "params",
            data_dir=root / "data",
            verbose=False,
        )


def test_run_compare_requires_cases_or_config() -> None:
    """run_compare は cases か compare_config のどちらかが必須."""
    with pytest.raises(ValidationError, match="--config または --cases"):
        run_compare(cases=None, compare_config=None)


def test_run_compare_from_cases(tmp_path: Path) -> None:
    """既存の case 出力から run_compare で result.png / comparison.png を生成する."""
    # 擬似 case_01 出力
    (tmp_path / "case_01").mkdir()
    (tmp_path / "case_01" / "shear_strain_for_plot.csv").write_text(
        "displacement_mm,theory_shear_strain,point,median\n0,0.0,0.0,0.0\n1,0.001,0.001,0.001\n"
    )
    (tmp_path / "case_01" / "metrics.csv").write_text(
        "kernel,rmse,max_error,mean_error,correlation,case_id\npoint,0.0,0.0,0.0,1.0,case_01\n"
    )
    (tmp_path / "case_02").mkdir()
    (tmp_path / "case_02" / "shear_strain_for_plot.csv").write_text(
        "displacement_mm,theory_shear_strain,point\n0,0.0,0.0\n1,0.001,0.001\n"
    )
    (tmp_path / "case_02" / "metrics.csv").write_text(
        "kernel,rmse,max_error,mean_error,correlation,case_id\npoint,0.01,0.01,0.0,0.99,case_02\n"
    )
    df, out_dir = run_compare(cases=["case_01", "case_02"], output_dir=tmp_path)
    assert out_dir == tmp_path / "compare_case_01_case_02"
    assert (out_dir / "result.png").exists()
    assert (out_dir / "comparison.png").exists()
    assert (out_dir / "metrics_compared.csv").exists()
    assert len(df) == 2
    assert list(df["case_id"]) == ["case_01", "case_02"]


def test_cmd_compare_with_config(tmp_path: Path) -> None:
    """compare サブコマンドを YAML 指定で実行する."""
    (tmp_path / "case_01").mkdir()
    (tmp_path / "case_01" / "shear_strain_for_plot.csv").write_text(
        "displacement_mm,theory_shear_strain,point\n0,0.0,0.0\n1,0.001,0.001\n"
    )
    (tmp_path / "case_01" / "metrics.csv").write_text(
        "kernel,rmse,max_error,mean_error,correlation,case_id\npoint,0.0,0.0,0.0,1.0,case_01\n"
    )
    (tmp_path / "compare.yml").write_text("cases: [case_01]\noutput_subdir: compare_one\n")
    args = Namespace(
        config=tmp_path / "compare.yml",
        group_id=None,
        cases=None,
        compare_dir=None,
        output_dir=tmp_path,
        config_dir=tmp_path,
        data_dir=tmp_path,
        verbose=False,
    )
    assert cmd_compare(args) == 0
    assert (tmp_path / "compare_one" / "result.png").exists()
    assert (tmp_path / "compare_one" / "comparison.png").exists()
