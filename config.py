"""学習用の小さな設定。

右側のコメントに「論文・公式（LFM2.5-VL-3B / LFM2-VL）」の値を並べている。
このソースはCPUで動く規模にするため、ほぼすべての数字を小さくしている。
"""
from dataclasses import dataclass, field


@dataclass
class VisionConfig:
    # SigLIP2 NaFlex（So400M）風のViTエンコーダ
    patch: int = 8             # 公式: 16px パッチ
    dim: int = 128             # 公式: 1152
    layers: int = 4            # 公式: 27
    heads: int = 4             # 公式: 16
    max_side: int = 64         # 公式: 1枚 512x512 まで（それ以上はタイル分割）
    max_pos: int = 16          # 位置埋め込みの基準グリッド（公式: 16x16 を補間して可変解像度に合わせる）


@dataclass
class TextEncConfig:
    # SigLIPの事前学習でだけ使うテキスト側エンコーダ（VLMの本番では使わない）
    dim: int = 128
    layers: int = 2
    heads: int = 4
    max_len: int = 16          # 公式SigLIP2: 64トークン


@dataclass
class ConnectorConfig:
    unshuffle: int = 2         # 公式: PixelUnshuffle 2x2（トークン数を1/4にする）
    hidden: int = 256          # 公式: 2層MLP（中間次元は公開情報では未確認）


@dataclass
class LMConfig:
    vocab: int = 0             # data.py の語彙数で上書きする（公式: 約128k）
    dim: int = 128             # 公式 LFM2.5-2.6B: 2048
    # 層の並び。"c" = ゲート付き短い畳み込みブロック、"a" = GQAアテンションブロック
    # 公式: 30層のうち 22層が畳み込み、8層がGQA（具体的な並び順はモデルの設定ファイルを参照）
    layout: str = "ccacca"
    conv_kernel: int = 3       # 公式: k = 3
    heads: int = 4             # 公式: クエリ32ヘッド
    kv_heads: int = 2          # 公式: KV 8グループ（GQA）
    ff_hidden: int = 256       # 公式: SwiGLU 10,752
    max_len: int = 160         # 公式: 32K コンテキスト


@dataclass
class Config:
    vision: VisionConfig = field(default_factory=VisionConfig)
    text_enc: TextEncConfig = field(default_factory=TextEncConfig)
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    lm: LMConfig = field(default_factory=LMConfig)
