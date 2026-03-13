#!/usr/bin/env python3
"""
窓幅（区間長）を横軸、RMSE を縦軸にしたグラフを描画する。
SDF 2m 5kPa-V, SDF 2m 20kPa-V, EF 2m V, EF 2m H の4系列。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 窓幅 [m]。0 = point（窓なし）
windows = np.array([0, 2, 4, 6, 8, 10, 12])

# 各ケースの RMSE（metrics_compared.csv より）
ef_2m_h = np.array([0.01760, 0.00667, 0.00342, 0.00183, 0.00098, 0.00059, 0.00065])
ef_2m_v = np.array([0.02990, 0.00842, 0.00366, 0.00282, 0.00233, 0.00211, 0.00194])
sdf_2m_5kpa_v = np.array([0.00792, 0.00526, 0.00338, 0.00274, 0.00270, 0.00238, 0.00224])
sdf_2m_20kpa_v = np.array([0.00869, 0.00341, 0.00217, 0.00173, 0.00175, 0.00179, 0.00192])

def main():
    out_dir = Path(__file__).resolve().parent.parent / "output" / "compare_EF2mH_point_vs_moving_average"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rmse_vs_window_2m_cases.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(windows, ef_2m_h, "o-", label="EF 2m Horizontal", markersize=8)
    ax.plot(windows, ef_2m_v, "s-", label="EF 2m Vertical", markersize=8)
    ax.plot(windows, sdf_2m_5kpa_v, "^-", label="SDF 2m 5kPa Vertical", markersize=8)
    ax.plot(windows, sdf_2m_20kpa_v, "d-", label="SDF 2m 20kPa Vertical", markersize=8)

    ax.set_xlabel("Window width (m)", fontsize=14)
    ax.set_ylabel("RMSE", fontsize=14)
    ax.set_xticks(windows)
    ax.set_xticklabels(["point", "2", "4", "6", "8", "10", "12"])
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="upper right", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
