"""エラーハンドリングのテスト. Python 3.12."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.cli import run_single_case
from src.config import Config
from src.data_loader import load_csv
from src.exceptions import DataLoadError, ValidationError


def test_missing_data_file(tmp_path: Path) -> None:
    """データファイルが存在しないケースで DataLoadError が発生する."""
    common = tmp_path / "common.yml"
    common.write_text("smoothing: { window_m: 0.1 }\nplot: {}\n")
    case_yml = tmp_path / "case.yml"
    case_yml.write_text("""
case_id: "x"
data_file: "nonexistent.csv"
interval: { start_m: 0, end_m: 1 }
theory_endpoint: [1, 0.01]
""")
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "case_x.yml").write_text(case_yml.read_text())
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # data_dir に nonexistent.csv はない
    out = tmp_path / "output"
    with pytest.raises((DataLoadError, FileNotFoundError, ValidationError)):
        run_single_case(
            "case_x",
            output_dir=out,
            config_dir=tmp_path,
            data_dir=data_dir,
            verbose=False,
        )


def test_invalid_yaml(tmp_path: Path) -> None:
    """不正な YAML で ValidationError が発生する."""
    common = tmp_path / "common.yml"
    common.write_text("smoothing: { window_m: 0.1 }\nplot: {}\n")
    case_bad = tmp_path / "case_bad.yml"
    case_bad.write_text("invalid: yaml: [unclosed\n")
    with pytest.raises(ValidationError):
        Config.load(common, case_bad)


def test_corrupted_csv(tmp_path: Path) -> None:
    """ヘッダーしかない破損CSVで load_csv が ValueError を起こす."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("BASE_FILE,0mm\n")  # データ行なし → empty after read
    with pytest.raises(ValueError, match="空"):
        load_csv(bad_csv)


def test_config_validate_missing_file(tmp_path: Path) -> None:
    """存在しない共通設定パスで ValidationError."""
    with pytest.raises(ValidationError, match="存在しません"):
        Config.load(tmp_path / "nonexistent.yml", tmp_path / "case.yml")
