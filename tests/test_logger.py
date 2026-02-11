"""logger モジュールの単体テスト."""

from __future__ import annotations

import logging
from pathlib import Path

from src.logger import log_environment, setup_logger


def test_setup_logger_console() -> None:
    """コンソールのみのロガーが作れる."""
    logger = setup_logger("test_console", log_file=None, console=True)
    assert logger.name == "test_console"
    assert logger.level == logging.INFO


def test_setup_logger_file(tmp_path: Path) -> None:
    """ファイル出力付きロガーでログが書き込まれる."""
    log_file = tmp_path / "run.log"
    logger = setup_logger("test_file", log_file=log_file, console=False)
    logger.info("test message")
    logger.handlers[0].flush()
    assert log_file.exists()
    assert "test message" in log_file.read_text()


def test_log_environment(tmp_path: Path) -> None:
    """log_environment で環境情報が記録される."""
    log_file = tmp_path / "env.log"
    logger = setup_logger("env", log_file=log_file, console=False)
    log_environment(logger)
    for h in logger.handlers:
        h.flush()
    text = log_file.read_text()
    assert "Python" in text
    assert "実行環境" in text or "=====" in text
