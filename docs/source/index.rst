.. 光ファイバひずみ解析ツール documentation master file.

光ファイバひずみ解析ツール
==========================

光ファイバがせん断変形を受けた際の、伸縮ひずみからせん断ひずみを算出する手法の精度検証パイプラインです。

概要
----

- 伸縮ひずみ ε を γ = √(1 + ε²) - 1 でせん断ひずみに換算
- YAML で解析区間・理論値を設定
- CLI で単一/全ケース実行、ログ・CSV 出力で再現性を確保

クイックスタート
----------------

.. code-block:: bash

   pip install -r requirements.txt
   python -m src.cli run --case case_sample

API リファレンス
---------------

.. toctree::
   :maxdepth: 2

   api
