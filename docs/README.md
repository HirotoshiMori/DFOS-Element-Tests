# 光ファイバひずみ解析ツール（学生向け）

光ファイバがせん断変形を受けた際に、計測された伸縮ひずみからせん断ひずみを算出する手法の精度検証パイプライン。Rayleigh/Brillouin 散乱光を用いた要素試験結果を対象に、YAML で設定した解析区間・理論値に基づき換算・評価・図化を行う。

## 概要

本ツールは、光ファイバセンサの伸縮ひずみ（ε）をせん断ひずみ（γ_s）に換算し、理論値との誤差（RMSE 等）を評価する。換算式は ε ≥ 0 のとき **γ_s = √((ε + 1)² − 1)**、ε < 0 のとき **γ_s = √(1 − (ε + 1)²)**。複数ケースを一括実行でき、結果は CSV と図で出力される。

## 特徴

- **point と複数カーネルの比較**: **point**（区間中央に最も近い1点の値）と **kernels_to_evaluate**（median, moving_average, gaussian, triangular, epanechnikov, hann）を区間中央を中心とした**一窓の代表値**で比較。グラフのタイトル・凡例にはケース YAML の **description** を使用（未設定時は case_id）。
- **YAML による設定管理**: 共通パラメータ（窓幅・グラフ）とケース別（ファイル・区間・理論値・description）を分離。
- **再現性確保**: 使用した設定を `output/<case>/params_used/` にコピー。
- **Google Colab 対応**: リポジトリを Drive に置き、ノートブックを Colab で開いてセルを実行するだけで利用可能。追加ライブラリはノートブック内の pip で導入される。

## 動作環境

- **Python 3.12+**
- 依存: NumPy, Pandas, Matplotlib, SciPy, PyYAML, tqdm（`requirements-colab.txt` または `requirements.txt`）

## インストール

### Google Colab（推奨）

1. **別の手順で** GitHub からリポジトリを clone し、**そのフォルダごと Google Drive に保存**する。
2. Drive 上で `notebooks/colab_quickstart.ipynb` を開き、**「Colab で開く」** で起動する。
3. 「1. 環境準備」の 1 つ目のセルで **Drive マウント** と **REPO_PATH**（Drive 上のリポジトリパス）を設定し、実行する。デフォルトは `/content/drive/MyDrive/DFOS-Element-Tests`（リポジトリをこのフォルダに置いている場合）。必要なら変更する。2 つ目のセルで **pip インストール**（`requirements-colab.txt`）のみ行う（ノートブック内では clone しない）。
4. 必要に応じて「2. データ・出力先の設定」でデータ・出力先を Drive の別フォルダに指定できる。

### ローカル（venv 推奨）

```bash
git clone https://github.com/your-repo/dfos-element-tests.git
cd dfos-element-tests
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-colab.txt
```

## クイックスタート

### 1. データ準備

`data/` フォルダに CSV を配置する。形式: 1列目が位置（m）、2列目以降が伸縮ひずみ（各変位レベル）。ヘッダー例: `BASE_FILE,0mm,1mm,...`

### 2. 設定ファイル作成

- `params/common.yml`: 窓幅・グラフ設定（既存をそのまま利用可）
- `params/cases/case_01.yml`: 解析対象ファイル名、区間（start_m, end_m）、理論せん断ひずみ（theory_shear_strain）、**description**（グラフのタイトル・凡例に表示する文言、任意）を編集

### 3. 実行

- **params/case_list.yml がある場合**: その中の `cases` 一覧が対象。**すべて**実行するか、**指定した id だけ**実行できる。
- **case_list.yml が無い場合**: `params/cases/*.yml` を従来どおり対象にする。

```bash
# 全ケース一括（プログレスバー付き）
python -m src.cli run --case all
# または省略（デフォルトが all）
python -m src.cli run

# 単一ケース（id または cases/<name>.yml の stem）
python -m src.cli run --case case_01

# 指定 id のみ複数実行（case_list 使用時）
python -m src.cli run --case EF_2m_H EF_2m_V
```

Colab のノートブックからは `!python -m src.cli run --case case_01` のように呼び出せる。

### 4. 結果確認

- `output/<case_id>/run.log`: 実行ログ
- `output/<case_id>/metrics.csv`: カーネル別の RMSE, 最大誤差, 相関係数等
- `output/<case_id>/params_used/`: 使用した YAML のコピー
- `output/<case_id>/shear_strain_for_plot.csv`: 図化用せん断ひずみ（理論値・カーネル別）
- `output/<case_id>/result.png`: 理論 vs カーネル別の比較グラフ（タイトルは description）
- `output/<case_id>/metrics_comparison.png`: カーネル別指標の比較図
- `output/<case_id>/center_position.yml`: 区間・代表点の位置情報
- 複数ケース比較: `python -m src.cli compare --cases case_01 case_03` で `output/compare_<name>/result.png`, `comparison.png`（凡例に description を使用）

## 設定ファイル

### params/common.yml

- **smoothing**: 窓幅は **window_m**（実距離 [m]）か **window_interval_ratio**（区間長の倍数）のどちらかで指定。このほか default_kernel, kernels_to_evaluate, gaussian_sigma_ratio
- **plot**: multi_case_shared_yscale（比較図で全サブプロットの縦軸を揃える）, **ylim**（理論 vs 予測グラフの縦軸範囲 [min, max]（%）。省略時は自動）, dpi, figure_size 等
- **conversion**: formula（現状は geometric 固定）
- **plot**: aspect_ratio, font_size, legend_location, dpi, figure_size

### params/case_list.yml（任意）

多くのケースを一つの YAML にまとめたいときは、`params/case_list.yml` を置く。**id のみ指定**で、`data_file: id.csv` と `output_subdir: id` が自動設定される。

```yaml
cases:
  - id: EF_2m_H
    description: "EF 2m Horizontal"
  - id: EF_2m_V
    description: "EF 2m Vertical"
```

- **id**（必須）: ケース識別子。省略時は `data_file = id + ".csv"`, `output_subdir = id`。必要なら **data_file** / **output_subdir** を上書き可能。
- **description**: グラフのタイトル・凡例に使用。
- 実行は **すべて**（`--case all` または省略）か **指定 id のみ**（`--case id1 id2 ...`）のどちらか。

`case_list.yml` が存在するときは、単一・一括ともここから id を解決する。存在しないときは従来どおり `params/cases/*.yml` を対象にする。

### params/cases/*.yml

- **case_id**, **description**: 識別用。**description** はグラフのタイトル・凡例に表示される（例: `"EF-4m-H"`）。
- **data_file**: `data/` からのファイル名
- **interval**: start_m, end_m（解析区間 [m]）
- **displacement_mm**: 変位レベル（例: [0,1,...,10]）
- **theory_shear_strain**: 各変位に対応する理論せん断ひずみ（無次元）

## ライセンス

MIT License

## 問い合わせ

リポジトリの Issue または管理者へ連絡してください。
