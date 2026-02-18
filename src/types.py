"""YAML設定およびデータ構造の型定義（TypedDict）. Python 3.12."""

from __future__ import annotations

from typing import TypedDict


class IntervalConfig(TypedDict):
    """解析区間の設定."""

    start_m: float
    end_m: float


class SmoothingConfig(TypedDict, total=False):
    """平滑化パラメータ. 窓長は window_m（実距離 [m]）か window_interval_ratio（区間長の倍数）のどちらかで指定."""

    window_m: float
    window_interval_ratio: float
    default_kernel: str
    kernels_to_evaluate: list[str]
    include_point: bool  # true のとき区間中央の1点（平滑化なし）も評価・グラフに含める
    gaussian_sigma_ratio: float


class ConversionConfig(TypedDict, total=False):
    """換算パラメータ."""

    formula: str


class PlotConfig(TypedDict, total=False):
    """グラフ描画パラメータ."""

    aspect_ratio: float
    font_size: int
    font_axis_label: int
    font_title: int
    font_legend: int
    font_tick_label: int  # 軸の目盛り数字（tick labels）のフォントサイズ
    legend_location: str
    dpi: int
    figure_size: list[float]
    line_width: float
    multi_case_shared_yscale: bool  # 比較図で全サブプロットの縦軸を揃える
    ylim: list[float]  # 理論 vs 予測グラフの縦軸範囲 [min, max]（%）。省略時は自動


class CommonParams(TypedDict, total=False):
    """共通パラメータ（params/common.yml）の構造."""

    smoothing: SmoothingConfig
    conversion: ConversionConfig
    plot: PlotConfig


class CaseParams(TypedDict, total=False):
    """ケース固有パラメータ（params/cases/*.yml）の構造."""

    case_id: str
    description: str
    data_file: str
    interval: IntervalConfig
    theory_endpoint: list[float]  # 最終点 [displacement_mm, theory_shear_strain]。原点と結ぶ線形で理論値を算出。displacement_mm は CSV 列ラベルから取得
