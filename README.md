# 光ファイバひずみ解析ツール

光ファイバせん断ひずみの換算・評価パイプライン。学生は Google Colab でノートブックを利用、開発はローカルで実施する想定です。

- **学生向け**: [docs/README.md](docs/README.md) — インストール・クイックスタート・設定（Colab 含む）
- **開発者向け**: [docs/DEVELOPER.md](docs/DEVELOPER.md) — テスト・コード品質・Sphinx・リポジトリ方針

## クイックリンク

- Colab: リポジトリを Google Drive に置き、`notebooks/colab_quickstart.ipynb` を Colab で開いてセルを上から実行（clone は別途実施）
- 単一ケース実行: `python -m src.cli run --case case_01`
- 全ケース: `python -m src.cli run --case all`

ライセンス: MIT
