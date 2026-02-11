"""theory モジュールのテスト."""

from __future__ import annotations

import pytest

from src.theory import compute_theory_shear_strain


def test_compute_theory_shear_strain_empty() -> None:
    assert compute_theory_shear_strain([], None, 0.0, 1.0) == []


def test_compute_theory_shear_strain_from_endpoint() -> None:
    # 線形: endpoint (10, 0.01) → 0mm→0, 5mm→0.005, 10mm→0.01
    got = compute_theory_shear_strain([0, 5, 10], [10.0, 0.01], 0.0, 1.0)
    assert got == [0.0, 0.005, 0.01]


def test_compute_theory_shear_strain_from_interval() -> None:
    # interval 4m → half=2m=2000mm, strain = mm/1000 / 2 = mm/2000
    # 0, 2, 4 mm → 0, 0.001, 0.002
    got = compute_theory_shear_strain([0.0, 2.0, 4.0], None, 0.0, 4.0)
    assert len(got) == 3
    assert got[0] == 0.0
    assert abs(got[1] - 0.001) < 1e-9
    assert abs(got[2] - 0.002) < 1e-9
