# lfm-vl-educational-small

Liquid AI の小型VLM **LFM2.5-VL**（LFM2-VL系）の仕組みを学ぶための、CPUで動く小さな実装。
本物の重みは使わず、図形の画像と短い説明文の合成データで、最初から小さく学習する。

## 全体の流れ

```
画像（縦横比そのまま）
  └─[1] ビジョンエンコーダ（SigLIP2 NaFlex風 ViT）  … パッチごとのベクトル
       └─[2] コネクタ（PixelUnshuffle 2×2 → 2層MLP） … 画像トークン（数は1/4）
            └─[3] 言語モデル LFM2（短い畳み込み ＋ GQA）… 文のトークンと1本の列にして、次のトークンを予測
```

| ファイル | 中身 |
|---|---|
| `config.py` | 小さな設定。コメントに公式の値を並べている |
| `data.py` | 図形の画像と説明文の合成データ、トークン列（`<image_start>` `<image>`×n `<image_end>`）の作り方 |
| `model.py` | VisionEncoder / SigLIP（事前学習） / Connector / ShortConv / GQA / LFM2 / LFMVL |
| `train.py` | stage0 SigLIP → stage1 コネクタだけ → stage2 全部 |
| `demo.py` | 学習した重みで説明文を生成 |

## 論文・公式との対応

| 部品 | 論文・公式（LFM2-VL / LFM2.5-VL-3B） | このソース |
|---|---|---|
| ビジョンエンコーダ | SigLIP2 NaFlex So400M（27層、1152次元、パッチ16） | 4層、128次元、パッチ8 |
| 可変解像度 | 縦横比を保つ。512×512までは1枚、それ以上は512×512タイル＋サムネイル | 縦横比を保つ。タイル分割なし（画像が小さいため） |
| 位置埋め込み | 基準グリッドを補間して使う | 同じ（16×16を双線形補間） |
| エンコーダの事前学習 | 学習済みSigLIP2を使う（シグモイド損失＋追加の損失） | シグモイド損失だけで小さく再現（stage0） |
| コネクタ | PixelUnshuffle（トークン数1/4）→ 2層MLP | 同じ構成 |
| 言語モデル | LFM2.5-2.6B：30層（短い畳み込み22＋GQA 8）、2048次元、32Kコンテキスト | 6層（畳み込み4＋GQA 2）、128次元 |
| 短い畳み込み | B・C・h~ に分けてゲート → 深さ方向の畳み込み（k=3）→ ゲート | 同じ |
| GQA | クエリ32ヘッド、KV 8グループ、RoPE、QK-Norm | 4ヘッド、KV 2グループ、RoPE、QK-Norm |
| 学習の段階 | ①コネクタだけ → ②全部（学習率比 5:5:1）→ ③SFT、最後に重みのマージ | stage1・stage2 のみ。マージなし |
| 損失 | 次のトークン予測（答えの部分） | 同じ |

## 簡略化した点と理由

- **重みもデータも自前の小さなもの**：本物は数GB・数百億トークン。CPUで仕組みを追うため。
- **SigLIPのプーリングは平均**：公式はアテンションプーリング。損失の形（シグモイド）を追うのが目的なので。
- **タイル分割なし**：画像が小さいので不要。タイル分割は記事の図で説明する。
- **生成時に毎回全体を計算し直す**：公式はKVキャッシュと畳み込みの状態を持ち回って速くする。コードを短くするため。
- **言語モデルのテキスト事前学習なし**：本物は先にテキストだけで大量に学習してある。

## 実行

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
python train.py
python demo.py
```

※ このリポジトリのコードは、構文チェックだけして、まだ実行していない。

## 参考

- LFM2 Technical Report（LFM2-VLの章を含む）：https://arxiv.org/abs/2511.23404
- モデルカード：https://huggingface.co/LiquidAI/LFM2.5-VL-3B
- SigLIP：https://arxiv.org/abs/2303.15343 ／ SigLIP 2：https://arxiv.org/abs/2502.14786
