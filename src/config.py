"""YAML設定の読み込み・検証・デフォルト値. Python 3.12."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

logger = logging.getLogger(__name__)

from src.exceptions import ValidationError
from src.data_loader import get_displacement_mm_from_csv
from src.theory import compute_theory_shear_strain
from src.types import (
    CaseParams,
    CommonParams,
    IntervalConfig,
    PlotConfig,
    SmoothingConfig,
)

# デフォルト値
DEFAULT_WINDOW_M = 0.15
DEFAULT_DPI = 300
DEFAULT_FIGURE_SIZE = (8.0, 5.0)


@dataclass
class CommonConfig:
    """共通設定（common.yml の型安全な表現）. 窓長は window_m（実距離 [m]）か window_interval_ratio（区間長の倍数）のどちらかで指定."""

    window_m: float | None  # 実距離 [m]。None のときは window_interval_ratio を使用
    window_interval_ratio: float | None  # 区間長の倍数。None のときは window_m を使用
    default_kernel: str
    kernels_to_evaluate: list[str]
    gaussian_sigma_ratio: float
    figure_dpi: int
    figure_size: tuple[float, float]
    plot: PlotConfig

    def get_window_m(self, interval_length_m: float) -> float:
        """区間長 [m] から有効な窓幅 [m] を返す。interval_length_m は end_m - start_m。"""
        if self.window_m is not None and self.window_m > 0:
            return self.window_m
        if self.window_interval_ratio is not None and self.window_interval_ratio > 0 and interval_length_m > 0:
            return interval_length_m * self.window_interval_ratio
        return float(DEFAULT_WINDOW_M)

    @classmethod
    def from_dict(cls, d: SmoothingConfig | CommonParams) -> "CommonConfig":
        """辞書から CommonConfig を構築. 窓長は window_m または window_interval_ratio のどちらか（両方あれば window_m を優先）。"""
        if isinstance(d, dict) and "smoothing" in d:
            sm = d["smoothing"] or {}
            plot_raw = d.get("plot") or {}
        else:
            sm = d if isinstance(d, dict) else {}
            plot_raw = {}

        kernels: list[str] = sm.get("kernels_to_evaluate") or [
            "median",
            "moving_average",
            "gaussian",
            "triangular",
            "epanechnikov",
            "hann",
        ]
        fig_size_raw = plot_raw.get("figure_size") or [8, 5]
        fig_size = (float(fig_size_raw[0]), float(fig_size_raw[1]))

        raw_window_m = sm.get("window_m")
        raw_ratio = sm.get("window_interval_ratio")
        if raw_window_m is not None and (isinstance(raw_window_m, (int, float)) and float(raw_window_m) > 0):
            window_m = float(raw_window_m)
            window_interval_ratio = None
        elif raw_ratio is not None and (isinstance(raw_ratio, (int, float)) and float(raw_ratio) > 0):
            window_m = None
            window_interval_ratio = float(raw_ratio)
        else:
            window_m = float(sm.get("window_m", DEFAULT_WINDOW_M))
            window_interval_ratio = None

        return cls(
            window_m=window_m,
            window_interval_ratio=window_interval_ratio,
            default_kernel=str(sm.get("default_kernel", "gaussian")),
            kernels_to_evaluate=kernels,
            gaussian_sigma_ratio=float(sm.get("gaussian_sigma_ratio", 0.25)),
            figure_dpi=int(plot_raw.get("dpi", DEFAULT_DPI)),
            figure_size=fig_size,
            plot=dict(plot_raw),
        )


@dataclass
class CaseConfig:
    """ケース設定（cases/*.yml の型安全な表現）."""

    case_id: str
    data_file: str
    interval: IntervalConfig
    displacement_mm: list[float]
    theory_shear_strain: list[float]
    theory_endpoint: list[float] | None
    description: str

    @classmethod
    def from_dict(
        cls,
        d: CaseParams | dict[str, Any],
        *,
        interval_settings: dict[str, IntervalConfig] | None = None,
    ) -> "CaseConfig":
        """辞書から CaseConfig を構築. displacement_mm は YAML では指定せず CSV 列ラベルから取得する."""
        data_file = str(d.get("data_file", ""))
        interval = d.get("interval") or {}
        start_m = float(interval.get("start_m", 0))
        end_m = float(interval.get("end_m", 0))

        # 区間が未設定または無効なとき、ファイル名から自動解決
        if (start_m == 0 and end_m == 0) and data_file and interval_settings:
            key = _parse_data_file_interval_key(data_file)
            if key and key in interval_settings:
                resolved = interval_settings[key]
                start_m = resolved["start_m"]
                end_m = resolved["end_m"]

        endpoint_raw = d.get("theory_endpoint")
        theory_endpoint: list[float] | None = None
        if endpoint_raw and len(endpoint_raw) == 2:
            theory_endpoint = [float(endpoint_raw[0]), float(endpoint_raw[1])]

        # displacement_mm は CSV から取得するためここでは空。theory_shear_strain も CSV 解決後に計算する。
        return cls(
            case_id=str(d.get("case_id", "")),
            data_file=data_file,
            interval={"start_m": start_m, "end_m": end_m},
            displacement_mm=[],
            theory_shear_strain=[],
            theory_endpoint=theory_endpoint,
            description=str(d.get("description", "")),
        )


@dataclass
class Config:
    """統合設定（共通 + ケース）. 単一ケース用."""

    common: CommonConfig
    case: CaseConfig

    def get_resolved_window_m(self) -> float:
        """現在ケースの区間長から有効な窓幅 [m] を返す."""
        start = self.case.interval.get("start_m", 0.0)
        end = self.case.interval.get("end_m", 0.0)
        interval_len = end - start
        return self.common.get_window_m(interval_len)

    @classmethod
    def load(
        cls,
        common_path: Path,
        case_path: Path,
        *,
        config_dir: Path | None = None,
    ) -> "Config":
        """YAML から共通設定とケース設定を読み込む. config_dir を渡すと interval_settings.yml から区間を自動解決する."""
        common_data = _load_yaml(common_path)
        case_data = _load_yaml(case_path)
        common = CommonConfig.from_dict(common_data)
        interval_settings: dict[str, IntervalConfig] = {}
        if config_dir is not None:
            try:
                interval_settings = _load_interval_settings(config_dir)
            except ValidationError:
                pass
        case = CaseConfig.from_dict(case_data, interval_settings=interval_settings or None)
        return cls(common=common, case=case)

    @classmethod
    def load_from_case_dict(
        cls,
        common_path: Path,
        case_dict: dict[str, Any],
        config_dir: Path,
    ) -> "Config":
        """共通設定とケース用辞書から Config を構築（case_list.yml の1要素用）。"""
        common_data = _load_yaml(common_path)
        common = CommonConfig.from_dict(common_data)
        interval_settings: dict[str, IntervalConfig] = {}
        try:
            interval_settings = _load_interval_settings(config_dir)
        except ValidationError:
            pass
        case = CaseConfig.from_dict(case_dict, interval_settings=interval_settings or None)
        return cls(common=common, case=case)

    def with_displacement_mm_from_csv(self, data_dir: Path) -> "Config":
        """CSV の列ラベルから displacement_mm を取得し、理論値を算出して返す新しい Config を返す（I/O: CSV 読込、計算: theory モジュール）。"""
        filepath = Path(self.case.data_file) if Path(self.case.data_file).is_absolute() else (data_dir / self.case.data_file)
        disp = get_displacement_mm_from_csv(filepath)
        theory = compute_theory_shear_strain(
            disp,
            self.case.theory_endpoint,
            self.case.interval["start_m"],
            self.case.interval["end_m"],
        )
        case = CaseConfig(
            case_id=self.case.case_id,
            data_file=self.case.data_file,
            interval=self.case.interval,
            displacement_mm=disp,
            theory_shear_strain=theory,
            theory_endpoint=self.case.theory_endpoint,
            description=self.case.description,
        )
        return Config(common=self.common, case=case)

    def validate(self) -> list[str]:
        """
        設定を検証する.

        Returns:
            エラーメッセージのリスト. 空なら正常.
        """
        errors: list[str] = []

        if self.common.window_m is None and self.common.window_interval_ratio is None:
            errors.append("smoothing に window_m または window_interval_ratio のどちらかを指定してください")
        elif self.common.window_m is not None and self.common.window_m <= 0:
            errors.append(f"window_m は 0 より大きくしてください: {self.common.window_m=}")
        elif self.common.window_interval_ratio is not None and self.common.window_interval_ratio <= 0:
            errors.append(f"window_interval_ratio は 0 より大きくしてください: {self.common.window_interval_ratio=}")

        if not self.case.data_file:
            errors.append("case に data_file がありません")
        if not self.case.displacement_mm:
            errors.append("case に displacement_mm がありません")
        if not self.case.theory_shear_strain:
            errors.append("case に theory_shear_strain がありません")
        if len(self.case.displacement_mm) != len(self.case.theory_shear_strain):
            errors.append(
                f"displacement_mm と theory_shear_strain の長さが一致しません: "
                f"{len(self.case.displacement_mm)=}, {len(self.case.theory_shear_strain)=}"
            )

        start = self.case.interval.get("start_m", 0)
        end = self.case.interval.get("end_m", 0)
        if start > end:
            errors.append(f"interval: start_m ({start}) > end_m ({end}) は無効です")

        # F-01: 有効窓幅が区間長を超える場合の検証
        interval_len = end - start
        if interval_len > 0:
            resolved = self.get_resolved_window_m()
            ratio = resolved / interval_len
            if ratio > 4.0:
                errors.append(
                    f"窓幅 ({resolved}) が区間長 ({interval_len}) の 4 倍を超えています。"
                    f" ratio={ratio:.2f}. 意図的でなければ window_m / window_interval_ratio または interval を確認してください。"
                )
            elif ratio > 1.0:
                logger.warning(
                    "窓幅 (%s) が区間長 (%s) を超えています (ratio=%.2f)。"
                    " 平滑化は区間外データに依存します。意図的か確認してください。",
                    resolved,
                    interval_len,
                    ratio,
                )

        return errors


def load_case_list(config_dir: Path) -> list[dict[str, Any]] | None:
    """
    case_list.yml を読み、ケースのリストを返す。ファイルが無いか不正なら None。

    id のみ指定した場合: data_file = id + ".csv", output_subdir = id に自動設定する。
    各要素は id（必須）, description, data_file, output_subdir, case_id, interval, theory_endpoint 等を指定可能。
    """
    path = config_dir / "case_list.yml"
    if not path.exists():
        return None
    try:
        data = _load_yaml(path)
    except ValidationError:
        return None
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) == 0:
        return None
    out: list[dict[str, Any]] = []
    for c in cases:
        if not isinstance(c, dict):
            continue
        id_val = c.get("id")
        if id_val is None:
            continue
        id_str = str(id_val).strip()
        row = dict(c)
        row["id"] = id_str
        row.setdefault("data_file", id_str + ".csv")
        row.setdefault("output_subdir", id_str)
        row.setdefault("case_id", id_str)
        out.append(row)
    return out if out else None


def get_case_by_id(case_list: list[dict[str, Any]], case_id: str) -> dict[str, Any] | None:
    """case_list から id が一致する要素を返す。"""
    for c in case_list:
        if str(c.get("id", "")).strip() == str(case_id).strip():
            return c
    return None


def get_case_description(config_dir: Path, case_id: str) -> str:
    """ケースの description を返す。case_list.yml を優先、なければ cases/<id>.yml。無い場合は ''。"""
    case_list = load_case_list(config_dir)
    if case_list:
        c = get_case_by_id(case_list, case_id)
        if c is not None:
            return str(c.get("description", ""))
    case_path = config_dir / "cases" / f"{case_id}.yml"
    if not case_path.exists():
        return ""
    try:
        data = _load_yaml(case_path)
        return str(data.get("description", ""))
    except ValidationError:
        return ""


def _load_yaml(path: Path) -> dict[str, Any]:
    """YAML ファイルを読み込む. 失敗時は ValidationError."""
    if not path.exists():
        raise ValidationError(f"設定ファイルが存在しません: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise ValidationError(f"YAML の読み込みに失敗しました: {path}") from e
    if not isinstance(data, dict):
        raise ValidationError(f"YAML のルートは辞書である必要があります: {path}")
    return cast(dict[str, Any], data)


def _parse_data_file_interval_key(data_file: str) -> str | None:
    """
    データファイル名から interval_settings のキーを組み立てる.
    区切りはアンダースコア (_) でもハイフン (-) でも可。

    例: "EF_4m_H.csv" / "EF-4m-H.csv" -> "EF_4m_Null_H"
         "SDF_2m_5kPa_V.csv" / "SDF-2m-5kPa-V.csv" -> "SDF_2m_5kPa_V"
         "Single-2m-H-B.csv" -> "Single_2m_Null_H_B" (4要素で3番目が H/V のときは Null を挿入)
    3要素 (meas_length_direction) のときは pressure を "Null" とする.
    4要素で3番目が H または V のときは meas_length_Null_direction_suffix とする.
    """
    stem = Path(data_file).stem
    for sep in ("_", "-"):
        parts = stem.split(sep)
        if len(parts) == 3:
            meas, length, direction = parts
            return f"{meas}_{length}_Null_{direction}"
        if len(parts) == 4:
            # SDF-2m-5kPa-V -> SDF_2m_5kPa_V / Single-2m-H-B -> Single_2m_Null_H_B
            if parts[2] in ("H", "V"):
                return f"{parts[0]}_{parts[1]}_Null_{parts[2]}_{parts[3]}"
            return "_".join(parts)
    return None


def _load_interval_settings(config_dir: Path) -> dict[str, IntervalConfig]:
    """params/interval_settings.yml を読み、キー -> {start_m, end_m} の辞書を返す. ファイルが無い場合は空辞書."""
    path = config_dir / "interval_settings.yml"
    if not path.exists():
        return {}
    data = _load_yaml(path)
    raw = data.get("intervals") or {}
    result: dict[str, IntervalConfig] = {}
    for key, val in raw.items():
        if isinstance(val, dict) and "start_m" in val and "end_m" in val:
            result[str(key)] = {
                "start_m": float(val["start_m"]),
                "end_m": float(val["end_m"]),
            }
    return result


def load_compare_config(
    path: Path,
    group_id: str | None = None,
) -> dict[str, object]:
    """
    比較用 YAML を読み、実行する1グループ分の cases と output_subdir を返す.

    - YAML に groups がある場合: group_id で該当グループを取得。group_id 省略時は先頭。
    - groups がなく cases のみの場合: その cases を返す。
    戻り値: {"cases": ["case_01", "case_02", ...], "output_subdir": str | None, "description": str | None}
    """
    data = _load_yaml(path)
    groups = data.get("groups")
    if groups:
        if not isinstance(groups, list) or len(groups) == 0:
            raise ValidationError(f"compare YAML に groups が空または不正です: {path}")
        if group_id is not None:
            found = [g for g in groups if isinstance(g, dict) and g.get("id") == group_id]
            if not found:
                raise ValidationError(f"group_id '{group_id}' が compare YAML にありません: {path}")
            g = found[0]
        else:
            g = groups[0] if isinstance(groups[0], dict) else {}
        cases = g.get("cases") or []
        output_subdir = g.get("id") or g.get("output_subdir")
        description = g.get("description")
    else:
        cases = data.get("cases") or []
        output_subdir = data.get("output_subdir") or data.get("id")
        description = data.get("description")
    if not cases:
        raise ValidationError(f"compare YAML に cases がありません: {path}")
    return {
        "cases": [str(c) for c in cases],
        "output_subdir": str(output_subdir) if output_subdir else None,
        "description": str(description) if description else None,
    }
