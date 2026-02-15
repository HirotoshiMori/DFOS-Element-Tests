"""CLI: run / validate サブコマンド. Python 3.12."""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.config import (
    Config,
    get_case_by_id,
    get_case_description,
    load_case_list,
    load_compare_config,
)
from src.converter import convert_columns_to_shear
from src.data_loader import POSITION_COLUMN, apply_smoothing_kernel, load_case
from src.evaluator import evaluate_all
from src.exceptions import AnalysisError, DataLoadError, ValidationError
from src.logger import log_environment, setup_logger
from src.reporter import (
    plot_case_result,
    plot_comparison,
    plot_metrics_comparison,
    plot_multi_case_results,
    plot_multi_case_results_overlay,
)


@dataclass
class CaseResult:
    """単一ケースの計算結果（I/O を行わない compute_case_pipeline の戻り値）."""

    predicted_by_kernel: dict[str, list[float]]
    metrics_rows: list[dict[str, object]]
    displacement_mm: list[float]
    theory_shear_strain: list[float]
    center_info: dict
    case_id: str
    description: str
    elapsed_sec: float
    n_plot: int


def _load_case_config(
    config_dir: Path,
    case_name: str,
    data_dir: Path,
) -> tuple[Config, str, Path | None, dict | None]:
    """
    設定を読み込み、検証済み Config と出力サブディレクトリ名・case_path・case_dict を返す。
    case_list.yml があればそこから id で検索、なければ params/cases/<case_name>.yml を読む。
    戻り値: (Config, output_subdir, case_path or None, case_dict or None)
    """
    common_path = config_dir / "common.yml"
    case_list = load_case_list(config_dir)
    if case_list:
        case_dict = get_case_by_id(case_list, case_name)
        if case_dict is not None:
            config = Config.load_from_case_dict(common_path, case_dict, config_dir)
            config = config.with_displacement_mm_from_csv(data_dir)
            errors = config.validate()
            if errors:
                raise ValidationError("設定エラー:\n" + "\n".join(errors))
            out_subdir = str(case_dict.get("output_subdir", case_name)).strip()
            return (config, out_subdir, None, case_dict)
    case_path = config_dir / "cases" / f"{case_name}.yml"
    if not case_path.exists():
        raise ValidationError(f"ケース設定が存在しません: {case_path}")
    config = Config.load(common_path, case_path, config_dir=config_dir)
    config = config.with_displacement_mm_from_csv(data_dir)
    errors = config.validate()
    if errors:
        raise ValidationError("設定エラー:\n" + "\n".join(errors))
    return (config, case_name, case_path, None)


def compute_case_pipeline(config: Config, data_dir: Path, case_name: str) -> CaseResult:
    """
    1 ケース分の計算のみ行う（ファイル・logger に触れない）。
    入力は Config と data_dir のみ。CaseResult を返す。
    """
    case_config = {
        "data_file": config.case.data_file,
        "interval": config.case.interval,
    }
    window_m = config.get_resolved_window_m()
    df = load_case(
        data_dir,
        case_config,
        expand_window_m=window_m,
    )
    strain_cols = [c for c in df.columns if c != POSITION_COLUMN]
    shear_df = convert_columns_to_shear(df, strain_cols)

    start_m = config.case.interval.get("start_m", 0.0)
    end_m = config.case.interval.get("end_m", 0.0)
    center_m = (start_m + end_m) / 2.0
    pos = df[POSITION_COLUMN].to_numpy(dtype=float)
    center_idx = int(np.argmin(np.abs(pos - center_m)))
    if center_idx >= len(shear_df):
        center_idx = len(shear_df) - 1
    if center_idx < 0:
        center_idx = 0
    center_position_m = float(pos[center_idx])
    half = window_m / 2.0
    center_info = {
        "interval_start_m": start_m,
        "interval_end_m": end_m,
        "interval_center_m": center_m,
        "center_index": center_idx,
        "center_position_m": center_position_m,
        "window_m": window_m,
        "window_start_m": center_position_m - half,
        "window_end_m": center_position_m + half,
    }

    kernels = (["point"] + list(config.common.kernels_to_evaluate)) if config.common.include_point else list(config.common.kernels_to_evaluate)
    predicted_by_kernel: dict[str, list[float]] = {}
    for k in kernels:
        if k == "point":
            sdf = shear_df
        else:
            try:
                df_smooth = apply_smoothing_kernel(
                    df,
                    strain_cols,
                    kernel=k,
                    window_m=window_m,
                    gaussian_sigma_ratio=config.common.gaussian_sigma_ratio,
                )
                sdf = convert_columns_to_shear(df_smooth, strain_cols)
            except (ValueError, ImportError):
                continue
        pred = [float(sdf[f"shear_{c}"].iloc[center_idx]) for c in strain_cols]
        predicted_by_kernel[k] = pred

    theory = config.case.theory_shear_strain
    elapsed_sec = 0.0  # 呼び出し元で計測するためここでは未使用
    metrics_rows: list[dict[str, object]] = []
    for k, pred in predicted_by_kernel.items():
        n_align = min(len(theory), len(pred))
        if n_align == 0:
            continue
        y_true = np.array(theory[:n_align], dtype=float)
        y_pred = np.array(pred[:n_align], dtype=float)
        m = evaluate_all(y_true, y_pred)
        metrics_rows.append({
            "kernel": k,
            "rmse": m["rmse"],
            "max_error": m["max_error"],
            "mean_error": m["mean_error"],
            "correlation": m["correlation"],
            "elapsed_sec": elapsed_sec,
            "case_id": config.case.case_id or case_name,
        })

    n_plot = min(
        len(config.case.displacement_mm),
        len(config.case.theory_shear_strain),
        max(len(p) for p in predicted_by_kernel.values()) if predicted_by_kernel else 0,
    )
    return CaseResult(
        predicted_by_kernel=predicted_by_kernel,
        metrics_rows=metrics_rows,
        displacement_mm=config.case.displacement_mm,
        theory_shear_strain=config.case.theory_shear_strain,
        center_info=center_info,
        case_id=config.case.case_id or case_name,
        description=config.case.description or "",
        elapsed_sec=elapsed_sec,
        n_plot=n_plot,
    )


def _parse_plot_ylim(plot_cfg: dict | None) -> tuple[float, float] | None:
    """plot.ylim を [min, max] から (min, max) に変換。無効なら None。"""
    if not plot_cfg:
        return None
    raw = plot_cfg.get("ylim")
    if not raw or not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        return (float(raw[0]), float(raw[1]))
    except (TypeError, ValueError):
        return None


def write_case_outputs(
    case_out: Path,
    case_result: CaseResult,
    common_path: Path,
    case_path: Path | None,
    case_name: str,
    case_dict: dict | None = None,
    plot_ylim: tuple[float, float] | None = None,
    plot_config: dict | None = None,
) -> None:
    """1 ケース分の出力ファイルを書き出す（center_position, metrics, plot CSV, 図, params_used）。case_path が None のときは case_dict を YAML で保存する。"""
    with open(case_out / "center_position.yml", "w", encoding="utf-8") as f:
        yaml.dump(case_result.center_info, f, default_flow_style=False, allow_unicode=True)

    params_copy = case_out / "params_used"
    params_copy.mkdir(exist_ok=True)
    shutil.copy(common_path, params_copy / "common.yml")
    if case_path is not None:
        shutil.copy(case_path, params_copy / f"{case_name}.yml")
    elif case_dict is not None:
        with open(params_copy / f"{case_name}.yml", "w", encoding="utf-8") as f:
            yaml.dump(case_dict, f, default_flow_style=False, allow_unicode=True)

    metrics_df = pd.DataFrame(case_result.metrics_rows)
    if not metrics_df.empty:
        metrics_df = metrics_df[["kernel", "rmse", "max_error", "mean_error", "correlation", "elapsed_sec", "case_id"]]
    metrics_df.to_csv(case_out / "metrics.csv", index=False)
    if not metrics_df.empty:
        plot_metrics_comparison(metrics_df, case_out / "metrics_comparison.png", plot_config=plot_config)

    if case_result.n_plot > 0 and case_result.predicted_by_kernel:
        plot_df = pd.DataFrame({
            "displacement_mm": case_result.displacement_mm[: case_result.n_plot],
            "theory_shear_strain": case_result.theory_shear_strain[: case_result.n_plot],
            **{k: p[: case_result.n_plot] for k, p in case_result.predicted_by_kernel.items()},
        })
        plot_df.to_csv(case_out / "shear_strain_for_plot.csv", index=False)
        plot_case_result(
            case_result.displacement_mm[: case_result.n_plot],
            case_result.theory_shear_strain[: case_result.n_plot],
            predicted_by_kernel=case_result.predicted_by_kernel,
            output_path=case_out / "result.png",
            title=case_result.description or f"Case {case_result.case_id}: Theory vs Predicted (kernels)",
            ylim=plot_ylim,
            plot_config=plot_config,
        )


def create_parser() -> argparse.ArgumentParser:
    """CLI パーサーを作成する."""
    parser = argparse.ArgumentParser(
        prog="dfos-eval",
        description="光ファイバせん断ひずみ換算の精度検証",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="単一または全ケースを実行")
    run_p.add_argument("--case", nargs="*", default=["all"], help="ケースID を0個以上指定。省略または all で全件、1件で単一、複数でその id のみ実行")
    run_p.add_argument("--config-dir", type=Path, default=Path("params"), help="設定ディレクトリ")
    run_p.add_argument("--data-dir", type=Path, default=Path("data"), help="データディレクトリ")
    run_p.add_argument("--output-dir", type=Path, default=Path("output"), help="出力ディレクトリ")
    run_p.add_argument("--verbose", "-v", action="store_true", help="詳細ログをコンソールに出力")

    val_p = sub.add_parser("validate", help="YAML 設定を検証")
    val_p.add_argument("--config", type=Path, default=Path("params/common.yml"), help="共通設定YAML")
    val_p.add_argument("--case", type=str, help="検証するケースYAML（例: params/cases/case_01.yml）")
    val_p.add_argument("--data-dir", type=Path, default=Path("data"), help="データディレクトリ（displacement_mm を CSV から取得する際に使用）")
    val_p.add_argument("--verbose", "-v", action="store_true", help="詳細出力")

    cmp_p = sub.add_parser("compare", help="指定した複数ケースの比較グラフ（result.png / comparison.png）を出力")
    cmp_p.add_argument("--config", type=Path, default=None, help="比較用 YAML（groups または cases を定義）")
    cmp_p.add_argument("--group-id", type=str, default=None, help="compare YAML の groups を使うときのグループ id")
    cmp_p.add_argument("--cases", type=str, nargs="*", default=None, help="比較する case_id のリスト（--config なし時は必須）")
    cmp_p.add_argument("--compare-dir", type=str, default=None, help="出力サブディレクトリ名（省略時は compare_<case_01>_<case_02>_...）")
    cmp_p.add_argument("--output-dir", type=Path, default=Path("output"), help="出力ルート")
    cmp_p.add_argument("--config-dir", type=Path, default=Path("params"), help="設定ディレクトリ")
    cmp_p.add_argument("--data-dir", type=Path, default=Path("data"), help="データディレクトリ")
    cmp_p.add_argument("--verbose", "-v", action="store_true", help="詳細ログ")

    return parser


def run_single_case(
    case_name: str,
    output_dir: Path,
    config_dir: Path = Path("params"),
    data_dir: Path = Path("data"),
    verbose: bool = False,
) -> list[dict[str, object]]:
    """
    単一ケースを実行する.

    Args:
        case_name: ケースID（params/cases/<case_name>.yml の stem）.
        output_dir: 出力先ディレクトリ.
        config_dir: 設定ディレクトリ.
        data_dir: データディレクトリ.
        verbose: 詳細ログをコンソールに出すか.

    Returns:
        全カーネル分の評価指標のリスト（run_all_cases で summary に集約するため）.
    """
    config, output_subdir, case_path, case_dict = _load_case_config(config_dir, case_name, data_dir)
    case_id = config.case.case_id or case_name
    case_out = output_dir / output_subdir
    case_out.mkdir(parents=True, exist_ok=True)
    log_file = case_out / "run.log"
    level = 20 if verbose else 30
    logger = setup_logger("cli", log_file=log_file, level=level, console=verbose)
    log_environment(logger)

    t0 = time.perf_counter()
    logger.info("ケース %s 実行開始", case_id)
    try:
        case_result = compute_case_pipeline(config, data_dir, output_subdir)
    except FileNotFoundError as e:
        raise DataLoadError(f"データ読み込み失敗: {e}") from e
    except Exception as e:
        raise DataLoadError(f"データ読み込み失敗: {e}") from e

    logger.info("データ読み込み: %s", config.case.data_file)
    elapsed = time.perf_counter() - t0
    case_result.elapsed_sec = elapsed
    for row in case_result.metrics_rows:
        row["elapsed_sec"] = elapsed
    logger.info("換算完了")
    logger.info("中心位置保存: %s (position_m=%.6f)", case_out / "center_position.yml", case_result.center_info["center_position_m"])
    logger.info("実行完了 (処理時間: %.2f 秒)", elapsed)
    if case_result.metrics_rows:
        logger.info(
            "RMSE (point): %.6f, MaxError: %.6f, Correlation: %.4f",
            case_result.metrics_rows[0].get("rmse", 0),
            case_result.metrics_rows[0].get("max_error", 0),
            case_result.metrics_rows[0].get("correlation", 0),
        )
    logger.info("設定を保存: %s", case_out / "params_used")
    logger.info("指標保存: %s", case_out / "metrics.csv")
    logger.info("指標比較図保存: %s", case_out / "metrics_comparison.png")
    logger.info("図化対象せん断ひずみ保存: %s", case_out / "shear_strain_for_plot.csv")
    logger.info("図を保存: %s", case_out / "result.png")

    common_path = config_dir / "common.yml"
    plot_ylim = _parse_plot_ylim(config.common.plot)
    write_case_outputs(case_out, case_result, common_path, case_path, output_subdir, case_dict=case_dict, plot_ylim=plot_ylim, plot_config=config.common.plot)
    return case_result.metrics_rows


def load_case_results_for_display(
    output_dir: Path,
    case_id: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    output_dir / case_id の metrics.csv と shear_strain_for_plot.csv を読み、
    表示用の (metrics_df, shear_df) を返す。無い場合は (None, None) の要素を返す。
    """
    case_out = output_dir / case_id
    metrics_path = case_out / "metrics.csv"
    shear_path = case_out / "shear_strain_for_plot.csv"
    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else None
    shear_df = pd.read_csv(shear_path) if shear_path.exists() else None
    return (metrics_df, shear_df)


def _read_case_plot_data(case_out_dir: Path, case_id: str) -> dict | None:
    """output/<case_id>/shear_strain_for_plot.csv を読み、plot_multi_case_results 用の dict を返す。"""
    csv_path = case_out_dir / "shear_strain_for_plot.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "displacement_mm" not in df.columns or "theory_shear_strain" not in df.columns:
        return None
    disp = df["displacement_mm"].astype(float).tolist()
    theory = df["theory_shear_strain"].astype(float).tolist()
    kernel_cols = [c for c in df.columns if c not in ("displacement_mm", "theory_shear_strain")]
    predicted_by_kernel: dict[str, list[float]] = {}
    for c in kernel_cols:
        predicted_by_kernel[c] = df[c].astype(float).tolist()
    return {
        "case_id": case_id,
        "displacement_mm": disp,
        "theory_shear_strain": theory,
        "predicted_by_kernel": predicted_by_kernel,
    }


def _ensure_case_results(
    case_id: str,
    output_dir: Path,
    config_dir: Path,
    data_dir: Path,
    verbose: bool,
) -> None:
    """既に結果（shear_strain_for_plot.csv と metrics.csv）があればスキップし、無い場合のみ run_single_case を実行する。"""
    case_out = output_dir / case_id
    if (case_out / "shear_strain_for_plot.csv").exists() and (case_out / "metrics.csv").exists():
        return
    run_single_case(case_id, output_dir, config_dir=config_dir, data_dir=data_dir, verbose=verbose)


def collect_compare_data(
    output_dir: Path,
    case_list: list[str],
) -> tuple[list[dict], list[pd.DataFrame]]:
    """比較用に各ケースの plot 用 dict と metrics DataFrame を収集する（I/O のみ、描画・保存は行わない）。"""
    cases_data: list[dict] = []
    metrics_dfs: list[pd.DataFrame] = []
    for cid in case_list:
        d = _read_case_plot_data(output_dir / cid, cid)
        if d:
            cases_data.append(d)
        mpath = output_dir / cid / "metrics.csv"
        if mpath.exists():
            df = pd.read_csv(mpath)
            if "case_id" not in df.columns:
                df["case_id"] = cid
            metrics_dfs.append(df)
    return (cases_data, metrics_dfs)


def write_compare_outputs(
    out_compare: Path,
    cases_data: list[dict],
    metrics_dfs: list[pd.DataFrame],
    shared_yscale: bool = True,
    plot_ylim: tuple[float, float] | None = None,
    plot_config: dict | None = None,
) -> pd.DataFrame | None:
    """比較結果の図と CSV を書き出し、結合した metrics DataFrame を返す。metrics_dfs が空なら None。"""
    if cases_data:
        plot_multi_case_results_overlay(
            cases_data,
            out_compare / "result.png",
            ylim=plot_ylim,
            plot_config=plot_config,
        )
    if not metrics_dfs:
        return None
    # ケース順で対応する description を metrics に付与
    for i, df in enumerate(metrics_dfs):
        if i < len(cases_data) and cases_data[i].get("description"):
            metrics_dfs[i] = df.assign(description=cases_data[i]["description"])
    combined = pd.concat(metrics_dfs, ignore_index=True)
    combined.to_csv(out_compare / "metrics_compared.csv", index=False)
    plot_comparison(combined, out_compare / "comparison.png", plot_config=plot_config)
    return combined


def run_compare(
    cases: list[str] | None = None,
    compare_config: Path | str | None = None,
    group_id: str | None = None,
    compare_dir: str | None = None,
    output_dir: Path = Path("output"),
    config_dir: Path = Path("params"),
    data_dir: Path = Path("data"),
    verbose: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """
    指定した複数ケースの比較を実行し、result.png / comparison.png を出力する.

    各ケースは output_dir/<case_id>/ に shear_strain_for_plot.csv と metrics.csv が
    既にあれば個別実行はスキップし、既存結果を使って比較のみ行う。
    cases と compare_config のどちらかは必須。compare_config を渡すと YAML から cases を取得する。
    戻り値: (比較対象ケースの metrics を結合した DataFrame, 出力ディレクトリ).
    """
    if compare_config is not None:
        path = Path(compare_config)
        if not path.is_absolute() and not path.exists():
            alt = config_dir / path.name
            if alt.exists():
                path = alt
        cfg = load_compare_config(path, group_id=group_id)
        case_list = cfg["cases"]
        if compare_dir is None and cfg.get("output_subdir"):
            compare_dir = cfg["output_subdir"]
    elif cases:
        case_list = list(cases)
    else:
        raise ValidationError("compare 実行には --config または --cases を指定してください")
    if not case_list:
        raise ValidationError("比較対象ケースがありません")
    if compare_dir is None:
        compare_dir = "compare_" + "_".join(case_list)
    out_compare = output_dir / compare_dir
    out_compare.mkdir(parents=True, exist_ok=True)
    log_file = out_compare / "run.log"
    level = 20 if verbose else 30
    logger = setup_logger("cli.compare", log_file=log_file, level=level, console=verbose)
    log_environment(logger)
    logger.info("比較開始: %s（計 %d ケース）", ", ".join(case_list), len(case_list))
    ran_any = False
    for cid in tqdm(case_list, desc="比較準備（ケース実行・結果収集）"):
        case_out = output_dir / cid
        had_results = (
            (case_out / "shear_strain_for_plot.csv").exists()
            and (case_out / "metrics.csv").exists()
        )
        _ensure_case_results(cid, output_dir, config_dir, data_dir, verbose)
        if had_results:
            logger.info("%s: 結果ありのためスキップ", cid)
        else:
            logger.info("%s: 実行しました", cid)
            ran_any = True
    if not ran_any:
        logger.info("全ケース結果済みのため、比較のみ実行します。")
    cases_data, metrics_dfs = collect_compare_data(output_dir, case_list)
    for d in cases_data:
        d["description"] = get_case_description(config_dir, d["case_id"])

    # グラフ化する kernel: compare 設定で指定されていればそれを使い、無ければ common.yml の kernels_to_evaluate
    kernels_to_plot: list[str] | None = None
    if compare_config is not None:
        kernels_to_plot = cfg.get("kernels")
    common_yml = config_dir / "common.yml"
    common_data: dict = {}
    if common_yml.exists():
        with open(common_yml, encoding="utf-8") as f:
            common_data = yaml.safe_load(f) or {}
    if kernels_to_plot is None:
        sm = common_data.get("smoothing") or {}
        ke = sm.get("kernels_to_evaluate") or []
        include_point = bool(sm.get("include_point", True))
        kernels_to_plot = (["point"] + [str(k) for k in ke]) if include_point else [str(k) for k in ke]
    if kernels_to_plot:
        for d in cases_data:
            by_k = d.get("predicted_by_kernel") or {}
            d["predicted_by_kernel"] = {k: by_k[k] for k in kernels_to_plot if k in by_k}
        for i, mdf in enumerate(metrics_dfs):
            if not mdf.empty and "kernel" in mdf.columns:
                metrics_dfs[i] = mdf[mdf["kernel"].astype(str).isin(kernels_to_plot)].copy()

    plot_cfg = common_data.get("plot") or {}
    shared_yscale = bool(plot_cfg.get("multi_case_shared_yscale", True))
    plot_ylim = _parse_plot_ylim(plot_cfg)
    combined = write_compare_outputs(
        out_compare, cases_data, metrics_dfs,
        shared_yscale=shared_yscale,
        plot_ylim=plot_ylim,
        plot_config=plot_cfg,
    )
    if combined is not None:
        logger.info("比較結果保存: result.png, comparison.png, metrics_compared.csv")
        return combined, out_compare
    logger.warning("比較対象の metrics が読み込めず、比較図は出力していません")
    return pd.DataFrame(), out_compare


def cmd_compare(args: argparse.Namespace) -> int:
    """compare サブコマンド."""
    try:
        df, out_compare = run_compare(
            cases=args.cases,
            compare_config=args.config,
            group_id=args.group_id,
            compare_dir=args.compare_dir,
            output_dir=args.output_dir,
            config_dir=args.config_dir,
            data_dir=args.data_dir,
            verbose=args.verbose,
        )
        print(f"比較結果: {out_compare}")
        if not df.empty:
            print("  result.png, comparison.png, metrics_compared.csv を保存しました")
        return 0
    except (ValidationError, DataLoadError, AnalysisError, FileNotFoundError) as e:
        print(f"エラー: {e}")
        return 1


def run_all_cases(
    output_dir: Path,
    config_dir: Path = Path("params"),
    data_dir: Path = Path("data"),
    case_ids: list[str] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    全ケース、または case_ids で指定したケースのみ実行する。
    case_list.yml がある場合はその id 一覧を使い、case_ids が None なら全件・指定時はその id のみ。
    case_list.yml が無い場合は params/cases/*.yml の glob で全件（case_ids は無視）。

    Returns:
        全ケースの評価結果をまとめた DataFrame.
    """
    case_list = load_case_list(config_dir)
    if case_list:
        names_to_run: list[str] = [str(c["id"]) for c in case_list] if case_ids is None else list(case_ids)
    else:
        cases_dir = config_dir / "cases"
        if not cases_dir.is_dir():
            raise ValidationError(f"ケース設定ディレクトリがありません: {cases_dir}")
        case_files = sorted(cases_dir.glob("*.yml"))
        if not case_files:
            raise ValidationError(f"ケースYAMLが1件もありません: {cases_dir}")
        names_to_run = [p.stem for p in case_files]

    rows: list[dict[str, float | str]] = []
    for case_name in tqdm(names_to_run, desc="ケース実行"):
        try:
            results = run_single_case(
                case_name,
                output_dir,
                config_dir=config_dir,
                data_dir=data_dir,
                verbose=verbose,
            )
            if isinstance(results, list):
                rows.extend(results)
            else:
                rows.append({k: v for k, v in results.items() if k != "method"})
        except (DataLoadError, ValidationError, AnalysisError, FileNotFoundError) as e:
            if verbose:
                raise
            rows.append({"case_id": case_name, "kernel": "", "error": str(e), "rmse": float("nan")})
    return pd.DataFrame(rows)


def cmd_validate(args: argparse.Namespace) -> int:
    """validate サブコマンド."""
    common_path = args.config
    if not common_path.exists():
        print(f"エラー: 設定ファイルが存在しません: {common_path}")
        return 1
    try:
        data = __import__("yaml").safe_load(common_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            print("エラー: YAML のルートは辞書である必要があります")
            return 1
        print(f"OK: {common_path}")
    except Exception as e:
        print(f"エラー: {e}")
        return 1

    if args.case:
        case_path = Path(args.case)
        if not case_path.exists():
            print(f"エラー: ケースファイルが存在しません: {case_path}")
            return 1
        config_dir = getattr(args, "config_dir", None) or case_path.parent.parent
        data_dir = getattr(args, "data_dir", None) or Path("data")
        try:
            config = Config.load(common_path, case_path, config_dir=config_dir)
            config = config.with_displacement_mm_from_csv(data_dir)
            config.validate()
            print(f"OK: {case_path}")
        except (ValidationError, FileNotFoundError) as e:
            print(f"検証エラー: {e}")
            return 1
    return 0


def main() -> int:
    """エントリポイント."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "validate":
        return cmd_validate(args)

    if args.command == "run":
        output_dir = args.output_dir
        config_dir = args.config_dir
        data_dir = args.data_dir
        verbose = args.verbose
        case_args = args.case if args.case else ["all"]
        run_all = len(case_args) == 0 or (len(case_args) == 1 and str(case_args[0]).strip().lower() == "all")

        def _common_plot_config(c_dir: Path) -> dict:
            common_yml = c_dir / "common.yml"
            if not common_yml.exists():
                return {}
            with open(common_yml, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("plot") or {}

        try:
            if run_all:
                df = run_all_cases(output_dir, config_dir=config_dir, data_dir=data_dir, case_ids=None, verbose=verbose)
                summary_path = output_dir / "summary.csv"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(summary_path, index=False)
                print(f"サマリー保存: {summary_path}")
                if not df.empty and "rmse" in df.columns:
                    plot_comparison(df, output_dir / "comparison.png", plot_config=_common_plot_config(config_dir))
                    print(f"比較グラフ保存: {output_dir / 'comparison.png'}")
                return 0
            if len(case_args) == 1:
                results = run_single_case(case_args[0], output_dir, config_dir=config_dir, data_dir=data_dir, verbose=verbose)
            else:
                df = run_all_cases(output_dir, config_dir=config_dir, data_dir=data_dir, case_ids=case_args, verbose=verbose)
                summary_path = output_dir / "summary.csv"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(summary_path, index=False)
                print(f"サマリー保存: {summary_path}")
                if not df.empty and "rmse" in df.columns:
                    plot_comparison(df, output_dir / "comparison.png", plot_config=_common_plot_config(config_dir))
                    print(f"比較グラフ保存: {output_dir / 'comparison.png'}")
                return 0
            # single case
            if results and isinstance(results, list):
                r0 = results[0]
                print(f"RMSE: {r0.get('rmse', 0):.6f}, MaxError: {r0.get('max_error', 0):.6f} (kernel: {r0.get('kernel', 'point')})")
                print("全カーネル比較は metrics.csv を参照")
            elif results:
                print(f"RMSE: {results.get('rmse', 0):.6f}, MaxError: {results.get('max_error', 0):.6f}")
            return 0
        except (ValidationError, DataLoadError, AnalysisError) as e:
            print(f"エラー: {e}")
            return 1

    if args.command == "compare":
        return cmd_compare(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
