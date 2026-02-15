"""結果集約・サマリーCSV・比較グラフ. Python 3.12."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Use Latin/English fonts only (no CJK) for consistent graph labels
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
except ImportError:
    plt = None  # type: ignore[assignment]


def aggregate_results(output_dir: Path) -> pd.DataFrame:
    """
    全ケースの結果を集約する.

    output_dir/summary.csv があればそれを読み、なければ
    output_dir/<case_id>/metrics.csv を走査して結合する.

    Returns:
        行=ケース、列=評価指標の DataFrame.
    """
    summary_path = output_dir / "summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)

    rows: list[pd.DataFrame] = []
    for case_dir in sorted(output_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        metrics_path = case_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        if "case_id" not in df.columns and df.shape[0] > 0:
            df["case_id"] = case_dir.name
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def create_summary_csv(results: pd.DataFrame, output_path: Path) -> None:
    """サマリーCSVを作成する."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)


# Marker/color cycle for multiple kernel curves (theory + up to N kernels)
_PLOT_STYLES = [
    ("o", 6),
    ("s", 5),
    ("^", 5),
    ("D", 5),
    ("v", 5),
    ("*", 7),
    ("P", 5),
    ("X", 5),
]


def _get_figsize(plot_cfg: dict | None, default: tuple[float, float] = (8, 5)) -> tuple[float, float]:
    """plot_config から figure_size を取得。無効なら default."""
    if not plot_cfg or "figure_size" not in plot_cfg:
        return default
    raw = plot_cfg["figure_size"]
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return (float(raw[0]), float(raw[1]))
    return default


def _get_dpi(plot_cfg: dict | None, default: int = 150) -> int:
    """plot_config から dpi を取得。無効なら default."""
    if not plot_cfg or "dpi" not in plot_cfg:
        return default
    try:
        return int(plot_cfg["dpi"])
    except (TypeError, ValueError):
        return default


def _get_fontsize(plot_cfg: dict | None, key: str, default: int) -> int:
    """plot_config からフォントサイズを取得。key は font_axis_label, font_title, font_legend。無ければ font_size、それも無ければ default."""
    if not plot_cfg:
        return default
    v = plot_cfg.get(key)
    if v is not None:
        try:
            return int(v)
        except (TypeError, ValueError):
            pass
    v = plot_cfg.get("font_size")
    if v is not None:
        try:
            return int(v)
        except (TypeError, ValueError):
            pass
    return default


def _get_line_width(plot_cfg: dict | None, default: float = 2.0) -> float:
    """plot_config から line_width を取得。無効なら default."""
    if not plot_cfg or "line_width" not in plot_cfg:
        return default
    try:
        return float(plot_cfg["line_width"])
    except (TypeError, ValueError):
        return default


def plot_case_result(
    displacement_mm: list[float],
    theory_shear_strain: list[float],
    predicted_shear_strain: list[float] | None = None,
    predicted_by_kernel: dict[str, list[float]] | None = None,
    output_path: Path | None = None,
    title: str = "Theory vs Predicted",
    ylim: tuple[float, float] | None = None,
    plot_config: dict | None = None,
) -> None:
    """
    Plot theoretical (line) and predicted (points) shear strain for a single case.
    If predicted_by_kernel is given, plot one series per kernel for comparison.
    X-axis: shear displacement (mm), Y-axis: shear strain (%).
    """
    if plt is None:
        raise ImportError("matplotlib が必要です")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    figsize = _get_figsize(plot_config, (7, 4))
    lw = _get_line_width(plot_config)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        displacement_mm,
        [t * 100 for t in theory_shear_strain],
        "k-",
        label="Theory",
        linewidth=lw,
    )
    n = len(displacement_mm)
    if predicted_by_kernel:
        for i, (kernel_name, pred) in enumerate(predicted_by_kernel.items()):
            p = pred[:n]
            if len(p) == 0:
                continue
            style, size = _PLOT_STYLES[i % len(_PLOT_STYLES)]
            ax.plot(
                displacement_mm[: len(p)],
                [x * 100 for x in p],
                marker=style,
                markersize=size,
                linestyle="",
                label=kernel_name,
            )
    elif predicted_shear_strain:
        p = predicted_shear_strain[:n]
        ax.plot(
            displacement_mm[: len(p)],
            [x * 100 for x in p],
            "ro",
            markersize=8,
            linestyle="",
            label="Predicted (mean)",
        )
    fs_axis = _get_fontsize(plot_config, "font_axis_label", 10)
    fs_title = _get_fontsize(plot_config, "font_title", 10)
    fs_legend = _get_fontsize(plot_config, "font_legend", 8)
    ax.set_xlabel("Shear displacement (mm)", fontsize=fs_axis)
    ax.set_ylabel("Shear strain (%)", fontsize=fs_axis)
    if ylim is not None and len(ylim) == 2:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_title(title, fontsize=fs_title)
    ax.legend(loc="best", fontsize=fs_legend)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=_get_dpi(plot_config))
    plt.close(fig)


def plot_multi_case_results(
    cases_data: list[dict],
    output_path: Path,
    ncols: int = 2,
    figsize_per_subplot: tuple[float, float] = (5.5, 3.5),
    shared_yscale: bool = True,
    ylim: tuple[float, float] | None = None,
    plot_config: dict | None = None,
) -> None:
    """
    複数ケースの理論 vs 予測をサブプロットで1枚にまとめる.

    cases_data: 各要素は case_id, displacement_mm, theory_shear_strain, predicted_by_kernel を持つ dict.
    shared_yscale: True のとき全サブプロットの縦軸スケールを揃える。
    ylim: 縦軸範囲 [min, max]（%）。指定時は全サブプロットに適用。
    """
    if plt is None:
        raise ImportError("matplotlib が必要です")
    if not cases_data:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_figsize = _get_figsize(plot_config, (8, 5))
    # figure_size をサブプロット数で分割する代わりに、1サブプロットあたりのサイズとして利用可能な比で算出
    sp_w = figsize_per_subplot[0] if not plot_config or "figure_size" not in plot_config else base_figsize[0] / ncols
    sp_h = figsize_per_subplot[1] if not plot_config or "figure_size" not in plot_config else base_figsize[1] / max(1, (len(cases_data) + ncols - 1) // ncols)
    n = len(cases_data)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(sp_w * ncols, sp_h * nrows),
        squeeze=False,
        sharey=shared_yscale,
    )
    lw = _get_line_width(plot_config)
    fs_axis = _get_fontsize(plot_config, "font_axis_label", 10)
    fs_title = _get_fontsize(plot_config, "font_title", 10)
    fs_legend = _get_fontsize(plot_config, "font_legend", 6)
    axes_flat = axes.ravel()
    for idx, case_dict in enumerate(cases_data):
        ax = axes_flat[idx]
        case_id = case_dict.get("case_id", f"case_{idx}")
        disp = case_dict.get("displacement_mm") or []
        theory = case_dict.get("theory_shear_strain") or []
        by_kernel = case_dict.get("predicted_by_kernel") or {}
        ax.plot(disp, [t * 100 for t in theory], "k-", label="Theory", linewidth=lw)
        n_pts = len(disp)
        for i, (kname, pred) in enumerate(by_kernel.items()):
            p = (pred[:n_pts] if hasattr(pred, "__getitem__") else []) or []
            if not p:
                continue
            style, size = _PLOT_STYLES[i % len(_PLOT_STYLES)]
            ax.plot(
                disp[: len(p)],
                [x * 100 for x in p],
                marker=style,
                markersize=size,
                linestyle="",
                label=kname,
            )
        ax.set_xlabel("Shear displacement (mm)", fontsize=fs_axis)
        ax.set_ylabel("Shear strain (%)", fontsize=fs_axis)
        if ylim is not None and len(ylim) == 2:
            ax.set_ylim(ylim[0], ylim[1])
        title = case_dict.get("description") or str(case_id)
        ax.set_title(title, fontsize=fs_title)
        ax.legend(loc="best", fontsize=fs_legend)
        ax.grid(True, alpha=0.3)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_get_dpi(plot_config))
    plt.close(fig)


def plot_multi_case_results_overlay(
    cases_data: list[dict],
    output_path: Path,
    figsize: tuple[float, float] = (8, 5),
    ylim: tuple[float, float] | None = None,
    plot_config: dict | None = None,
) -> None:
    """
    複数ケースの理論 vs 予測を1枚のグラフに重ねて描画する.
    ケースごとに色を分け、各ケースの theory（線）と kernel 予測（マーカー）を同じ色で表示する.
    """
    if plt is None:
        raise ImportError("matplotlib が必要です")
    if not cases_data:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figsize = _get_figsize(plot_config, figsize)
    lw = _get_line_width(plot_config)
    fig, ax = plt.subplots(figsize=figsize)
    # ケース数に応じた色（理論線・予測点をケースごとに同一色で区別）
    base = list(plt.cm.tab10.colors) if hasattr(plt.cm, "tab10") else list(plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"]))
    while len(base) < len(cases_data) and hasattr(plt.cm, "tab20"):
        base.extend(plt.cm.tab20.colors)
    colors = (base * (1 + len(cases_data) // max(1, len(base))))[: len(cases_data)]
    for idx, case_dict in enumerate(cases_data):
        case_id = case_dict.get("case_id", f"case_{idx}")
        desc = case_dict.get("description") or case_id
        disp = case_dict.get("displacement_mm") or []
        theory = case_dict.get("theory_shear_strain") or []
        by_kernel = case_dict.get("predicted_by_kernel") or {}
        color = colors[idx % len(colors)]
        n_pts = len(disp)
        ax.plot(
            disp,
            [t * 100 for t in theory],
            "-",
            color=color,
            linewidth=lw,
            label=f"Theory ({desc})",
        )
        for ki, (kname, pred) in enumerate(by_kernel.items()):
            p = (pred[:n_pts] if hasattr(pred, "__getitem__") else []) or []
            if not p:
                continue
            style, size = _PLOT_STYLES[ki % len(_PLOT_STYLES)]
            ax.plot(
                disp[: len(p)],
                [x * 100 for x in p],
                marker=style,
                markersize=size,
                linestyle="",
                color=color,
                label=f"{desc} {kname}",
            )
    fs_axis = _get_fontsize(plot_config, "font_axis_label", 10)
    fs_title = _get_fontsize(plot_config, "font_title", 10)
    fs_legend = _get_fontsize(plot_config, "font_legend", 7)
    ax.set_xlabel("Shear displacement (mm)", fontsize=fs_axis)
    ax.set_ylabel("Shear strain (%)", fontsize=fs_axis)
    if ylim is not None and len(ylim) == 2:
        ax.set_ylim(ylim[0], ylim[1])
    ax.legend(loc="best", fontsize=fs_legend)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_get_dpi(plot_config))
    plt.close(fig)


def plot_metrics_comparison(metrics_df: pd.DataFrame, output_path: Path, plot_config: dict | None = None) -> None:
    """
    metrics.csv のカーネル別指標を比較図化する（どれがよいか一目で分かるようにする）.
    RMSE / max_error / mean_error は小さいほどよい、correlation は大きいほどよい。
    """
    if plt is None:
        raise ImportError("matplotlib が必要です")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dpi = _get_dpi(plot_config)
    fs_axis = _get_fontsize(plot_config, "font_axis_label", 10)
    fs_title = _get_fontsize(plot_config, "font_title", 10)
    if metrics_df.empty or "kernel" not in metrics_df.columns:
        fig, ax = plt.subplots()
        ax.set_title("Metrics comparison (no data)", fontsize=fs_title)
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        return

    kernels = metrics_df["kernel"].astype(str).tolist()
    n = len(kernels)
    if n == 0:
        return
    x = range(n)
    figsize = _get_figsize(plot_config, (9, 7))
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # RMSE (lower is better)
    ax = axes[0, 0]
    vals = metrics_df["rmse"].to_numpy() if "rmse" in metrics_df.columns else [0] * n
    colors = ["#2ecc71" if v == min(vals) else "steelblue" for v in vals]
    ax.bar(x, vals, color=colors, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(kernels, rotation=45, ha="right", fontsize=fs_axis)
    ax.set_ylabel("RMSE", fontsize=fs_axis)
    ax.set_title("RMSE (lower is better)", fontsize=fs_title)
    ax.grid(True, alpha=0.3, axis="y")

    # Max error (lower is better)
    ax = axes[0, 1]
    vals = metrics_df["max_error"].to_numpy() if "max_error" in metrics_df.columns else [0] * n
    colors = ["#2ecc71" if v == min(vals) else "steelblue" for v in vals]
    ax.bar(x, vals, color=colors, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(kernels, rotation=45, ha="right", fontsize=fs_axis)
    ax.set_ylabel("Max error", fontsize=fs_axis)
    ax.set_title("Max error (lower is better)", fontsize=fs_title)
    ax.grid(True, alpha=0.3, axis="y")

    # Mean error (lower abs is better; here show as-is)
    ax = axes[1, 0]
    vals = metrics_df["mean_error"].to_numpy() if "mean_error" in metrics_df.columns else [0] * n
    best_idx = int(np.argmin(np.abs(vals)))
    colors = ["#2ecc71" if i == best_idx else "steelblue" for i in range(n)]
    ax.bar(x, vals, color=colors, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(kernels, rotation=45, ha="right", fontsize=fs_axis)
    ax.set_ylabel("Mean error", fontsize=fs_axis)
    ax.set_title("Mean error (|smaller| is better)", fontsize=fs_title)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")

    # Correlation (higher is better)
    ax = axes[1, 1]
    vals = metrics_df["correlation"].to_numpy() if "correlation" in metrics_df.columns else [0] * n
    colors = ["#2ecc71" if v == max(vals) else "steelblue" for v in vals]
    ax.bar(x, vals, color=colors, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(kernels, rotation=45, ha="right", fontsize=fs_axis)
    ax.set_ylabel("Correlation", fontsize=fs_axis)
    ax.set_title("Correlation (higher is better)", fontsize=fs_title)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_comparison(results: pd.DataFrame, output_path: Path, plot_config: dict | None = None) -> None:
    """
    ケース間比較グラフを作成する.

    各ケースの RMSE を棒グラフで表示する.
    """
    if plt is None:
        raise ImportError("matplotlib が必要です")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dpi = _get_dpi(plot_config)
    fs_axis = _get_fontsize(plot_config, "font_axis_label", 10)
    fs_title = _get_fontsize(plot_config, "font_title", 10)
    fs_legend = _get_fontsize(plot_config, "font_legend", 7)

    if results.empty or "rmse" not in results.columns:
        fig, ax = plt.subplots()
        ax.set_title("Comparison (no data)", fontsize=fs_title)
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        return

    if "description" in results.columns and "kernel" in results.columns:
        labels = [f"{d}_{k}" if k else str(d) for d, k in zip(results["description"].astype(str), results["kernel"].astype(str))]
    elif "description" in results.columns:
        labels = results["description"].astype(str).tolist()
    elif "case_id" in results.columns and "kernel" in results.columns:
        labels = [f"{c}_{k}" if k else str(c) for c, k in zip(results["case_id"].astype(str), results["kernel"].astype(str))]
    elif "case_id" in results.columns:
        labels = results["case_id"].astype(str).tolist()
    else:
        labels = results.index.astype(str).tolist()
    figsize = _get_figsize(plot_config, (max(8, len(labels) * 0.4), 4))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(results)), results["rmse"], color="steelblue", edgecolor="black")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs_legend)
    ax.set_ylabel("RMSE", fontsize=fs_axis)
    ax.set_xlabel("Case" + (" / Kernel" if "kernel" in results.columns else ""), fontsize=fs_axis)
    ax.set_title("RMSE by case" + (" and kernel" if "kernel" in results.columns else ""), fontsize=fs_title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
