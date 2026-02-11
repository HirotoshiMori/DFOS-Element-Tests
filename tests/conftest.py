"""pytest 共通フィクスチャ・サンプルデータパス."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def data_dir() -> Path:
    """プロジェクトルートの data ディレクトリ."""
    return Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def fixtures_dir() -> Path:
    """tests/fixtures ディレクトリ."""
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def sample_csv_path(fixtures_dir: Path) -> Path:
    """サンプルCSV（5行×3列）のパス."""
    p = fixtures_dir / "sample.csv"
    if not p.exists():
        pytest.skip("tests/fixtures/sample.csv がありません")
    return p
