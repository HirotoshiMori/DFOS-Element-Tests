"""CSV読み込み・区間抽出・前処理・平滑化. Python 3.12."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.types import CaseParams

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    gaussian_filter1d = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

# 定数（マジックナンバー禁止）
POSITION_COLUMN_LEGACY = "BASE_FILE"
POSITION_COLUMN = "position"
DEFAULT_SIGMA_THRESHOLD = 3.0
# F-06: 伸縮ひずみ |ε| がこの値を超える点数を WARNING する閾値（無次元）
STRAIN_ABS_WARNING_THRESHOLD = 1.0


def _warn_strain_outliers(df: pd.DataFrame, strain_cols: list[str], context: str = "") -> None:
    """|ε| > STRAIN_ABS_WARNING_THRESHOLD のデータ点数をカウントし、1 件以上なら WARNING を出す。"""
    if not strain_cols:
        return
    stack = df[strain_cols].abs()
    over = (stack > STRAIN_ABS_WARNING_THRESHOLD).sum().sum()
    if over > 0:
        logger.warning(
            "伸縮ひずみ |ε| > %.1f のデータが %d 件あります。%s",
            STRAIN_ABS_WARNING_THRESHOLD,
            int(over),
            context or "データ",
        )


def load_csv(filepath: Path) -> pd.DataFrame:
    """CSVファイルを読み込む。

    1列目を位置[m]、2列目以降を伸縮ひずみ[ɛ]として解釈する。
    ヘッダーは「BASE_FILE」「0mm」「1mm」等を想定。

    Args:
        filepath: CSVファイルのパス。

    Returns:
        行=位置[m]、列=ひずみのDataFrame。1列目は position にリネーム済み。

    Raises:
        FileNotFoundError: ファイルが存在しない。
        ValueError: CSV形式が不正（空、数値列なし等）。
    """
    if not filepath.exists():
        raise FileNotFoundError(f"入力ファイルが存在しません: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"CSVの読み込みに失敗しました: {filepath}") from e

    if df.empty:
        raise ValueError(f"CSVが空です: {filepath}")

    # 1列目を位置にリネーム（BASE_FILE または先頭列名）
    first_col = df.columns[0]
    if first_col == POSITION_COLUMN_LEGACY or first_col != POSITION_COLUMN:
        df = df.rename(columns={first_col: POSITION_COLUMN})

    # 位置列を数値に
    df[POSITION_COLUMN] = pd.to_numeric(df[POSITION_COLUMN], errors="coerce")
    if df[POSITION_COLUMN].isna().any():
        raise ValueError(f"位置列に非数値が含まれています: {filepath}")

    # ひずみ列（2列目以降）を数値に
    strain_cols = [c for c in df.columns if c != POSITION_COLUMN]
    if not strain_cols:
        raise ValueError(f"ひずみ列がありません: {filepath}")

    for col in strain_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(POSITION_COLUMN).reset_index(drop=True)
    _warn_strain_outliers(df, strain_cols, f"CSV全体 ({filepath.name})")
    logger.info("CSV読み込み完了: %s, 行数=%d, ひずみ列数=%d", filepath.name, len(df), len(strain_cols))
    return df


# 列ラベルから「数字+mm」を抽出する正規表現（例: 0mm, 2.0mm, ..._2.0mm[ini]_...）
_RE_MM_NUMBER = re.compile(r"(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)


def _parse_mm_from_column_name(col: str) -> float | None:
    """列名から 'mm' の直前にくる数値（整数または小数）を返す。見つからなければ None。"""
    m = _RE_MM_NUMBER.search(str(col))
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def get_displacement_mm_from_csv(filepath: Path) -> list[float]:
    """CSV のひずみ列ラベルから displacement_mm [mm] のリストを取得する。

    位置列以外の列名を左から順にパースする。
    - 「0mm」「1mm」「2.0mm」のように、mm の直後の数値（整数または小数）を抽出。
    - 「20250925_4m_vertical_n2_0mm[ini]_FD_CH1.fdd」のような長いラベルでも、
      mm の前の数字（ここでは 0）をピックアップする。

    Args:
        filepath: CSV ファイルのパス。

    Returns:
        変位 [mm] のリスト（列の並び順）。2.0 のような表記は float で保持。
    """
    if not filepath.exists():
        raise FileNotFoundError(f"入力ファイルが存在しません: {filepath}")
    df_header = pd.read_csv(filepath, nrows=0)
    cols = df_header.columns.tolist()
    if not cols:
        return []
    strain_cols = cols[1:]  # 1列目は位置
    result: list[float] = []
    for c in strain_cols:
        val = _parse_mm_from_column_name(str(c))
        if val is not None:
            result.append(val)
        else:
            result.append(float(len(result)))  # フォールバック: 連番
    return result


def extract_interval(
    df: pd.DataFrame,
    start_m: float,
    end_m: float,
) -> pd.DataFrame:
    """指定区間 [start_m, end_m] の行のみ抽出する。

    範囲がデータ外の場合は警告し、利用可能範囲でクリップする。

    Args:
        df: position 列を持つ DataFrame。
        start_m: 区間開始位置 [m]。
        end_m: 区間終了位置 [m]。

    Returns:
        区間内の行のみの DataFrame。
    """
    if POSITION_COLUMN not in df.columns:
        raise ValueError("DataFrame に position 列がありません")

    pos = df[POSITION_COLUMN]
    actual_min, actual_max = float(pos.min()), float(pos.max())

    if start_m > actual_max or end_m < actual_min:
        logger.warning(
            "解析区間 [%.2f, %.2f] がデータ範囲 [%.2f, %.2f] と重なりません。空になる可能性があります。",
            start_m,
            end_m,
            actual_min,
            actual_max,
        )

    start_clip = max(start_m, actual_min)
    end_clip = min(end_m, actual_max)
    if start_clip != start_m or end_clip != end_m:
        logger.warning(
            "区間をデータ範囲内にクリップしました: [%.2f, %.2f] -> [%.2f, %.2f]",
            start_m,
            end_m,
            start_clip,
            end_clip,
        )

    mask = (df[POSITION_COLUMN] >= start_clip) & (df[POSITION_COLUMN] <= end_clip)
    out = df.loc[mask].copy().reset_index(drop=True)

    if out.empty:
        logger.warning("抽出区間内にデータがありません。空の DataFrame を返します。")

    logger.info("区間抽出: [%.2f, %.2f] -> %d 行", start_clip, end_clip, len(out))
    return out


def preprocess(
    df: pd.DataFrame,
    interpolate_missing: bool = True,
    remove_outliers: bool = False,
    sigma_threshold: float = DEFAULT_SIGMA_THRESHOLD,
) -> pd.DataFrame:
    """データ前処理（欠損値補間・外れ値除外オプション）。

    仕様上、欠損値は線形補間。外れ値は context では「除外しない」のため
    remove_outliers のデフォルトは False。

    Args:
        df: 入力データ（position 列 + ひずみ列）。
        interpolate_missing: 欠損値を線形補間するか。
        remove_outliers: 外れ値を除外するか（±sigma_threshold σ）。
        sigma_threshold: 外れ値判定の閾値（標準偏差の倍数）。

    Returns:
        前処理済み DataFrame。

    Raises:
        ValueError: 空データ、または全欠損の列がある場合。
    """
    if df.empty:
        raise ValueError("前処理対象の DataFrame が空です")

    strain_cols = [c for c in df.columns if c != POSITION_COLUMN]
    if not strain_cols:
        raise ValueError("ひずみ列がありません")

    out = df.copy()

    # 欠損値補間（全欠損列は補間不可のため除外してから補間）
    if interpolate_missing:
        missing_per_col = out[strain_cols].isna().sum()
        all_missing_cols = missing_per_col[missing_per_col == len(out)].index.tolist()
        if all_missing_cols:
            logger.warning("全欠損のひずみ列を除外します（線形補間不可）: %s", ", ".join(all_missing_cols))
            strain_cols = [c for c in strain_cols if c not in all_missing_cols]
            out = out.drop(columns=all_missing_cols, errors="ignore")
            if not strain_cols:
                raise ValueError("有効なひずみ列が残りません（全列が欠損でした）")
        total_missing = int(out[strain_cols].isna().sum().sum()) if strain_cols else 0
        if total_missing > 0:
            out[strain_cols] = out[strain_cols].interpolate(method="linear", limit_direction="both")
            logger.info("欠損値を線形補間しました: 合計 %d 件", total_missing)
        else:
            logger.info("欠損値はありません")
    else:
        remaining = out[strain_cols].isna().sum().sum()
        if remaining > 0:
            logger.warning("欠損値補間をスキップしました。残り欠損: %d 件", int(remaining))

    # 外れ値除外（オプション）
    if remove_outliers and strain_cols:
        before_len = len(out)
        for col in strain_cols:
            mu = out[col].mean()
            std = out[col].std()
            if std == 0 or pd.isna(std):
                continue
            low = mu - sigma_threshold * std
            high = mu + sigma_threshold * std
            out = out[(out[col] >= low) & (out[col] <= high)]
        removed = before_len - len(out)
        if removed > 0:
            logger.warning("外れ値除外: %d 行を削除（閾値 ±%.1f σ）", removed, sigma_threshold)
        if out.empty:
            raise ValueError("外れ値除外後にデータが空になりました")
    else:
        logger.info("外れ値除外は行いません（remove_outliers=False）")

    return out.reset_index(drop=True)


def _window_points_from_meters(df: pd.DataFrame, window_m: float) -> int:
    """position 列の間隔から窓幅 [m] を点数に換算する. 奇数にする."""
    if POSITION_COLUMN not in df.columns or len(df) < 2:
        return 1
    pos = df[POSITION_COLUMN].to_numpy(dtype=float)
    spacing = float(np.median(np.diff(pos)))
    if spacing <= 0:
        return 1
    w = max(1, int(round(window_m / spacing)))
    return w if w % 2 == 1 else w + 1


def apply_smoothing_kernel(
    df: pd.DataFrame,
    strain_columns: list[str],
    kernel: str,
    window_m: float,
    gaussian_sigma_ratio: float = 0.25,
) -> pd.DataFrame:
    """
    ひずみ列に沿って指定カーネルで平滑化する.
    位置は position 列で、等間隔でない場合は窓幅を点数に換算して適用する.

    Args:
        df: position 列 + ひずみ列の DataFrame.
        strain_columns: 平滑化対象の列名リスト.
        kernel: "median", "moving_average", "gaussian", "triangular", "epanechnikov", "hann".
        window_m: 窓幅 [m].
        gaussian_sigma_ratio: gaussian 時の σ/窓幅 比率.

    Returns:
        position 列はそのまま、ひずみ列のみ平滑化した DataFrame.
    """
    out = df.copy()
    win = _window_points_from_meters(df, window_m)
    # 窓がデータ点数を超えると rolling が NaN になるため、データ長でキャップ（奇数にそろえる）
    n = len(df)
    if win > n:
        win = n if n % 2 == 1 else max(1, n - 1)
    min_periods = max(1, win // 2)

    for col in strain_columns:
        if col not in out.columns:
            continue
        s = out[col]
        if kernel == "median":
            out[col] = s.rolling(window=win, center=True, min_periods=min_periods).median()
        elif kernel == "moving_average":
            out[col] = s.rolling(window=win, center=True, min_periods=min_periods).mean()
        elif kernel == "gaussian":
            if gaussian_filter1d is None:
                raise ImportError("gaussian カーネルには scipy が必要です")
            sigma = max(0.5, win * gaussian_sigma_ratio)
            out[col] = gaussian_filter1d(s.to_numpy(dtype=float), sigma=sigma, mode="nearest")
        elif kernel == "triangular":
            out[col] = (
                s.rolling(window=win, center=True, min_periods=min_periods, win_type="triang")
                .mean()
            )
        elif kernel == "epanechnikov":
            # 0.75 * (1 - u^2), u in [-1, 1]
            u = np.linspace(-1, 1, win)
            w_full = 0.75 * np.maximum(0, 1 - u**2)

            def weighted_mean(x: np.ndarray) -> float:
                if np.any(np.isnan(x)):
                    return np.nan
                n = len(x)
                # F-02: 端部でも重みを中心寄せで対称にスライスする
                offset = (len(w_full) - n) // 2
                w = w_full[offset : offset + n].copy()
                w /= w.sum()
                return float(np.average(x, weights=w))

            out[col] = s.rolling(window=win, center=True, min_periods=min_periods).apply(
                weighted_mean, raw=True
            )
        elif kernel == "hann":
            out[col] = (
                s.rolling(window=win, center=True, min_periods=min_periods, win_type="hann")
                .mean()
            )
        else:
            raise ValueError(f"未対応のカーネル: {kernel}")

    return out


def load_case(
    data_dir: Path,
    case_config: CaseParams,
    expand_window_m: float | None = None,
) -> pd.DataFrame:
    """ケース設定に従い CSV を読み、区間抽出・前処理まで実行する。

    責務: I/O（load_csv）と計算（extract_interval, preprocess）を一括で実行。
    呼び出し元は data_dir と case_config のみを渡す。

    Args:
        data_dir: データディレクトリ（CSV の親）。
        case_config: ケースYAMLの内容（data_file, interval 等）。
        expand_window_m: 窓幅 [m]。指定時は区間を両側に expand_window_m/2 だけ広げて抽出し、
            窓が区間より大きい場合に区間外データも使う。

    Returns:
        前処理済み DataFrame。

    Raises:
        FileNotFoundError: ファイルが存在しない。
        ValueError: 設定不正または前処理でエラー。
    """
    data_file = case_config.get("data_file")
    if not data_file:
        raise ValueError("case_config に data_file がありません")

    filepath = Path(data_file) if Path(data_file).is_absolute() else (data_dir / data_file)
    df = load_csv(filepath)

    interval = case_config.get("interval")
    if interval is not None:
        start_m = float(interval["start_m"])
        end_m = float(interval["end_m"])
        if expand_window_m is not None and expand_window_m > 0:
            half = expand_window_m / 2.0
            start_m = start_m - half
            end_m = end_m + half
        df = extract_interval(df, start_m, end_m)
        strain_cols_here = [c for c in df.columns if c != POSITION_COLUMN]
        _warn_strain_outliers(df, strain_cols_here, "抽出区間内")

    df = preprocess(df, interpolate_missing=True, remove_outliers=False)
    return df
