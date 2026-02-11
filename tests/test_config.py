"""config モジュールの単体テスト."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    CaseConfig,
    CommonConfig,
    Config,
    get_case_by_id,
    load_case_list,
    load_compare_config,
    _parse_data_file_interval_key,
)
from src.exceptions import ValidationError


def test_common_config_from_dict() -> None:
    """CommonConfig.from_dict でデフォルトが入る."""
    empty: dict = {}
    c = CommonConfig.from_dict(empty)
    assert c.window_m == 0.15
    assert c.window_interval_ratio is None
    assert c.default_kernel == "gaussian"
    assert "median" in c.kernels_to_evaluate


def test_common_config_window_interval_ratio() -> None:
    """窓幅を区間長の倍数で指定したとき get_window_m が区間長×比率を返す."""
    c = CommonConfig.from_dict({"smoothing": {"window_interval_ratio": 2.0}})
    assert c.window_m is None
    assert c.window_interval_ratio == 2.0
    assert c.get_window_m(4.0) == 8.0
    assert c.get_window_m(0.5) == 1.0


def test_case_config_from_dict() -> None:
    """CaseConfig.from_dict で辞書から構築（displacement_mm は CSV から取得するためここでは空）。"""
    d = {
        "case_id": "01",
        "data_file": "sample.csv",
        "interval": {"start_m": 0.0, "end_m": 1.0},
        "theory_endpoint": [1, 1e-5],
    }
    c = CaseConfig.from_dict(d)
    assert c.case_id == "01"
    assert c.data_file == "sample.csv"
    assert c.interval["start_m"] == 0.0
    assert c.displacement_mm == []
    assert c.theory_shear_strain == []
    assert c.theory_endpoint == [1.0, 1e-5]


def test_config_load_and_validate(tmp_path: Path) -> None:
    """Config.load で YAML 読み込み、CSV から displacement_mm 取得後に validate."""
    common = tmp_path / "common.yml"
    common.write_text("""
smoothing:
  window_m: 0.1
  default_kernel: gaussian
plot:
  dpi: 300
""")
    case = tmp_path / "case.yml"
    case.write_text("""
case_id: "t1"
data_file: "x.csv"
interval: { start_m: 0, end_m: 1 }
theory_endpoint: [1, 0.01]
""")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.csv").write_text("BASE_FILE,0mm,1mm\n0.0,0.0,0.01\n")
    config = Config.load(common, case)
    config = config.with_displacement_mm_from_csv(tmp_path / "data")
    errors = config.validate()
    assert errors == []
    assert config.case.displacement_mm == [0, 1]
    assert config.case.theory_shear_strain == [0.0, 0.01]


def test_config_validate_errors(tmp_path: Path) -> None:
    """interval 不正で validate がエラーを返す（displacement_mm 未解決のままでも interval エラーが出る）."""
    common = tmp_path / "common.yml"
    common.write_text("smoothing: { window_m: 0.1 }\nplot: {}\n")
    case = tmp_path / "case.yml"
    case.write_text("""
case_id: "t1"
data_file: "x.csv"
interval: { start_m: 2, end_m: 1 }
theory_endpoint: [1, 0.0]
""")
    config = Config.load(common, case)
    errors = config.validate()
    assert any("start_m" in e or "end_m" in e for e in errors)


def test_config_load_missing_file(tmp_path: Path) -> None:
    """存在しないケースYAMLで ValidationError."""
    common = tmp_path / "common.yml"
    common.write_text("smoothing: { window_m: 0.1 }\nplot: {}\n")
    with pytest.raises(ValidationError, match="存在しません"):
        Config.load(common, tmp_path / "nonexistent.yml")


def test_parse_data_file_interval_key() -> None:
    """データファイル名から interval_settings キーを組み立てる。区切りは _ でも - でも可."""
    assert _parse_data_file_interval_key("EF_4m_H.csv") == "EF_4m_Null_H"
    assert _parse_data_file_interval_key("SDF_2m_5kPa_V.csv") == "SDF_2m_5kPa_V"
    assert _parse_data_file_interval_key("DF_2m_V") == "DF_2m_Null_V"
    assert _parse_data_file_interval_key("EF-4m-H.csv") == "EF_4m_Null_H"
    assert _parse_data_file_interval_key("EF-2m-H") == "EF_2m_Null_H"
    assert _parse_data_file_interval_key("SDF-2m-5kPa-V.csv") == "SDF_2m_5kPa_V"
    assert _parse_data_file_interval_key("Single-2m-H-B.csv") == "Single_2m_Null_H_B"
    assert _parse_data_file_interval_key("Single-4m-H-B") == "Single_4m_Null_H_B"
    assert _parse_data_file_interval_key("x") is None
    assert _parse_data_file_interval_key("a_b_c_d_e") is None


def test_interval_auto_from_settings(tmp_path: Path) -> None:
    """interval 未指定時、data_file 名と interval_settings.yml から区間を自動解決する."""
    (tmp_path / "common.yml").write_text("smoothing: { window_m: 0.1 }\nplot: {}\n")
    (tmp_path / "interval_settings.yml").write_text("""
intervals:
  EF_4m_Null_H: { start_m: 8.78, end_m: 12.78 }
""")
    (tmp_path / "case.yml").write_text("""
case_id: "01"
data_file: "EF_4m_H.csv"
theory_endpoint: [1, 0.01]
""")
    config = Config.load(tmp_path / "common.yml", tmp_path / "case.yml", config_dir=tmp_path)
    assert config.case.interval["start_m"] == 8.78
    assert config.case.interval["end_m"] == 12.78


def test_theory_shear_strain_auto_from_interval(tmp_path: Path) -> None:
    """theory_endpoint なし時、CSV から displacement_mm 取得後 displacement(m)/(interval(m)/2) で理論値を算出."""
    (tmp_path / "common.yml").write_text("smoothing: { window_m: 0.1 }\nplot: {}\n")
    (tmp_path / "case.yml").write_text("""
case_id: "01"
data_file: "x.csv"
interval: { start_m: 8.0, end_m: 12.0 }
""")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.csv").write_text("BASE_FILE,0mm,2mm,4mm\n0.0,0,0,0\n")
    config = Config.load(tmp_path / "common.yml", tmp_path / "case.yml")
    config = config.with_displacement_mm_from_csv(tmp_path / "data")
    # interval_m = 4, half = 2. theory = (mm/1000)/2 = mm/2000 → 0, 0.001, 0.002
    assert config.case.displacement_mm == [0, 2, 4]
    assert config.case.theory_shear_strain == [0.0, 0.001, 0.002]


def test_theory_shear_strain_from_endpoint(tmp_path: Path) -> None:
    """theory_endpoint [disp_mm, strain] 指定時、CSV から displacement_mm 取得後原点と結ぶ線形で理論値を算出."""
    (tmp_path / "common.yml").write_text("smoothing: { window_m: 0.1 }\nplot: {}\n")
    (tmp_path / "case.yml").write_text("""
case_id: "01"
data_file: "x.csv"
interval: { start_m: 0, end_m: 1 }
theory_endpoint: [10, 0.005]
""")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.csv").write_text("BASE_FILE,0mm,5mm,10mm\n0.0,0,0,0\n")
    config = Config.load(tmp_path / "common.yml", tmp_path / "case.yml")
    config = config.with_displacement_mm_from_csv(tmp_path / "data")
    assert config.case.displacement_mm == [0, 5, 10]
    # 直線: strain = 0.005 * (disp / 10) → 0, 0.0025, 0.005
    assert config.case.theory_shear_strain == [0.0, 0.0025, 0.005]


def test_load_compare_config_single_group(tmp_path: Path) -> None:
    """比較用 YAML（groups なし、cases のみ）を読み込む."""
    (tmp_path / "compare.yml").write_text("""
cases: [case_01, case_02]
description: "2 cases"
""")
    cfg = load_compare_config(tmp_path / "compare.yml")
    assert cfg["cases"] == ["case_01", "case_02"]
    assert cfg.get("output_subdir") is None


def test_load_compare_config_groups(tmp_path: Path) -> None:
    """比較用 YAML（groups あり）を読み込み、group_id で1件取得."""
    (tmp_path / "groups.yml").write_text("""
groups:
  - id: g1
    cases: [case_01]
  - id: g2
    cases: [case_01, case_02]
    output_subdir: "my_compare"
""")
    cfg = load_compare_config(tmp_path / "groups.yml", group_id="g2")
    assert cfg["cases"] == ["case_01", "case_02"]
    # id が output_subdir より優先される
    assert cfg.get("output_subdir") == "g2"
    cfg1 = load_compare_config(tmp_path / "groups.yml", group_id="g1")
    assert cfg1["cases"] == ["case_01"]
    assert cfg1.get("output_subdir") == "g1"


def test_load_case_list_none(tmp_path: Path) -> None:
    """case_list.yml が無いとき load_case_list は None."""
    assert load_case_list(tmp_path) is None


def test_load_case_list_success(tmp_path: Path) -> None:
    """case_list.yml があれば cases のリストを返す。id のみなら data_file=id.csv, output_subdir=id に自動設定."""
    (tmp_path / "case_list.yml").write_text("""
cases:
  - id: EF_2m_H
    description: "EF 2m H"
  - id: two
    output_subdir: out_two
    description: "Second"
""")
    lst = load_case_list(tmp_path)
    assert lst is not None
    assert len(lst) == 2
    assert lst[0]["id"] == "EF_2m_H"
    assert lst[0]["data_file"] == "EF_2m_H.csv"
    assert lst[0]["output_subdir"] == "EF_2m_H"
    assert lst[1]["id"] == "two"
    assert lst[1]["data_file"] == "two.csv"
    assert lst[1]["output_subdir"] == "out_two"
    assert get_case_by_id(lst, "EF_2m_H") == lst[0]
    assert get_case_by_id(lst, "two") == lst[1]
    assert get_case_by_id(lst, "missing") is None
