"""理論せん断ひずみの計算（純粋計算）. Python 3.12."""

from __future__ import annotations


def compute_theory_shear_strain(
    displacement_mm: list[float],
    theory_endpoint: list[float] | None,
    start_m: float,
    end_m: float,
) -> list[float]:
    """
    theory_endpoint または interval から理論せん断ひずみリストを算出する（I/O なし）.

    Args:
        displacement_mm: 変位 [mm] のリスト.
        theory_endpoint: 最終点 [displacement_mm, theory_shear_strain]. 原点と結ぶ線形で補間.
        start_m: 区間開始 [m].
        end_m: 区間終了 [m].

    Returns:
        各変位に対応する理論せん断ひずみのリスト.
    """
    if not displacement_mm:
        return []
    if theory_endpoint and len(theory_endpoint) == 2:
        disp_end, strain_end = float(theory_endpoint[0]), float(theory_endpoint[1])
        if disp_end != 0:
            return [strain_end * (float(x) / disp_end) for x in displacement_mm]
    if end_m > start_m:
        interval_m = end_m - start_m
        half_interval = interval_m / 2.0
        return [(float(x) / 1000.0) / half_interval for x in displacement_mm]
    return []
