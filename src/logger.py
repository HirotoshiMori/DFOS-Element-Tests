"""ログのセットアップ・実行環境記録. Python 3.12."""

from __future__ import annotations

import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def setup_logger(
    name: str,
    log_file: Path | None = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    ロガーをセットアップする.

    Args:
        name: ロガー名.
        log_file: ログファイルパス. None の場合はファイル出力なし.
        level: ログレベル.
        console: True ならコンソールにも出力.

    Returns:
        設定済み Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # 既存ハンドラを消して重複を防ぐ
    logger.handlers.clear()

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_environment(logger: logging.Logger) -> None:
    """
    実行環境情報をログに記録する.

    記録内容: Python バージョン, OS, 依存ライブラリバージョン, 実行日時.
    """
    logger.info("===== 実行環境 =====")
    logger.info("Python: %s", sys.version.split()[0])
    logger.info("OS: %s (%s)", platform.system(), platform.machine())
    logger.info("実行日時: %s", datetime.now(timezone.utc).isoformat())

    for mod_name in ("numpy", "pandas", "matplotlib", "scipy", "yaml"):
        try:
            if mod_name == "yaml":
                import yaml as y
                ver = getattr(y, "__version__", "?")
            else:
                mod = __import__(mod_name)
                ver = getattr(mod, "__version__", "?")
            logger.info("%s: %s", mod_name.capitalize(), ver)
        except ImportError:
            logger.info("%s: (未インストール)", mod_name.capitalize())
    logger.info("==================")
