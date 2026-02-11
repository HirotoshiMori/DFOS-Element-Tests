# Sphinx configuration for DFOS-Element-tests API docs.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

project = "光ファイバひずみ解析ツール"
copyright = "DFOS-Element-tests"
author = "DFOS-Element-tests"
release = "0.1.0"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
templates_path = ["_templates"]
exclude_patterns: list[str] = []
language = "ja"
html_theme = "alabaster"
html_static_path = ["_static"]
html_theme_options = {"sidebar_width": "300px"}

# Napoleon for Google-style docstrings
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
