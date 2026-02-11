# 開発者向けドキュメント

本リポジトリは学生が Google Colab で利用することを想定して GitHub に push する。学生は GitHub から clone を別プロセスで行いリポジトリを Drive に保管したうえで、Drive 上のノートブックを Colab で開いて実行する。コードの改変は開発環境でのみ行い、テスト・CI・ドキュメントビルドも開発環境で実行する。学生向けの使い方は [README.md](README.md) を参照。

## リポジトリ構成

- **src/**: 本体（cli, config, converter, data_loader, evaluator, reporter, theory 等）
- **params/**: 共通・ケース別・比較用 YAML
- **notebooks/**: Colab クイックスタート
- **tests/**: pytest 用テスト
- **docs/**: 本ドキュメント・学生向け README・Sphinx 用 source/

## 開発環境の構築

```bash
git clone https://github.com/your-repo/dfos-element-tests.git
cd dfos-element-tests
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# または uv: uv sync
```

- 学生向けには `requirements-colab.txt`（実行に必要なパッケージのみ）を使用。開発時は `requirements.txt` で pytest, mypy, ruff, black, isort も導入する。

## テスト

```bash
pytest tests/ -v
# カバレッジ付き（htmlcov/ にレポート）
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

- テストは開発環境でのみ実行する。学生は Colab でノートブックのみ使用するため、テストの実行は不要。

## コード品質

```bash
ruff check src/ tests/
black src/ tests/
mypy src/ --python-version 3.12
# 必要に応じて isort
```

- `pyproject.toml` に mypy / pytest の設定あり。

## API ドキュメント（Sphinx）

```bash
pip install sphinx sphinx-rtd-theme
cd docs && make html
# ブラウザで docs/_build/html/index.html を開く
```

- `docs/_build/` は .gitignore で除外済み。ビルド成果物はコミットしない。

## .gitignore の方針

- **コミットしない**: `.venv/`, `__pycache__/`, `data/`, `output/`, `.pytest_cache/`, `htmlcov/`, `.coverage`, `docs/_build/`, `.github/`, `.claude/`, `.idea/`, `.vscode/` 等。
- 学生が clone したときに含まれるもの: `src/`, `params/`, `notebooks/`, `tests/`, `docs/`（README.md, DEVELOPER.md, Sphinx の source 等）, `README.md`（ルート）, `requirements.txt`, `requirements-colab.txt`, `pyproject.toml`。学生は主にノートブックと params を触り、テストや CI は実行しない想定。

## 主な機能の実装メモ

- **グラフのタイトル・凡例**: ケース YAML の `description` を使用。`CaseResult.description`, `get_case_description(config_dir, case_id)`（config.py）, reporter の `plot_case_result` / `plot_multi_case_results` / `plot_comparison` で反映。
- **compare**: `run_compare` で `cases_data` に description を付与し、`write_compare_outputs` で metrics に description 列を追加してから `plot_comparison` に渡している。

## ライセンス

MIT License
