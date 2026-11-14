"""LFM2.5-VL を学ぶための小さな実装。

全体の流れ（公式と同じ並び）：
  画像 → [1] ビジョンエンコーダ（SigLIP2 NaFlex風のViT）→ パッチごとのベクトル
       → [2] コネクタ（PixelUnshuffle → 2層MLP）→ 言語モデルと同じ次元の「画像トークン」
       → [3] 言語モデル（ゲート付き短い畳み込み ＋ GQA のハイブリッド）→ 次のトークンの確率

[0] として、SigLIPの事前学習（画像と文を対応させるシグモイド損失）も入れている。
公式はSigLIP2の学習済み重みを使うので、この段階は自分では学習しない。
このソースでは「SigLIPがどう学習されたか」を追うために、小さく再現している。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 共通の部品
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """LFM2の言語モデルは全層でRMSNorm（平均を引かず、二乗平均の平方根で割るだけ）を使う。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """位置ごとのMLP。公式LFM2もSwiGLU（片方をSiLUに通し、もう片方とかけ算するゲート）。"""

    def __init__(self, dim, hidden):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def rope(x, pos):
    """RoPE：ベクトルを2次元ずつ組にして、位置に応じた角度だけ回す。
    x: (B, H, T, D)、pos: (T,)。common/position_encoding/rope の図解と同じ処理。"""
    d = x.shape[-1]
    freq = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device).float() / d))
    ang = pos[:, None].float() * freq[None]            # (T, D/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


# ---------------------------------------------------------------------------
# [1] ビジョンエンコーダ（SigLIP2 NaFlex風）
# ---------------------------------------------------------------------------
class ViTBlock(nn.Module):
    """普通のTransformerブロック（LayerNorm → 自己アテンション → 残差 → LayerNorm → MLP → 残差）。
    画像側は因果マスクなし：どのパッチも全パッチを見られる。"""

    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(approximate="tanh"), nn.Linear(dim * 4, dim))

    def forward(self, x, pad_mask=None):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class VisionEncoder(nn.Module):
    """画像を 縦横比を保ったまま パッチに切り、パッチごとのベクトルを返す。

    NaFlex のポイント：
      - 画像を正方形に引き伸ばさない（縦横比を保つ）。パッチの数は画像ごとに違う。
      - 位置埋め込みは 基準グリッド（公式16×16）を学習しておき、画像のパッチ数（縦gh×横gw）に
        双線形補間で引き伸ばして使う。これで、どんな縦横のパッチ数にも位置の情報をつけられる。
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Linear(3 * cfg.patch * cfg.patch, cfg.dim)   # 公式もパッチを平らにして線形変換
        self.pos = nn.Parameter(torch.randn(1, cfg.dim, cfg.max_pos, cfg.max_pos) * 0.02)
        self.blocks = nn.ModuleList([ViTBlock(cfg.dim, cfg.heads) for _ in range(cfg.layers)])
        self.ln_post = nn.LayerNorm(cfg.dim)

    def patchify(self, img):
        """(3, H, W) → (gh*gw, 3*p*p)。左上から右へ、行ごとに並べる。"""
        p = self.cfg.patch
        c, h, w = img.shape
        gh, gw = h // p, w // p
        x = img.reshape(c, gh, p, gw, p).permute(1, 3, 0, 2, 4).reshape(gh * gw, c * p * p)
        return x, gh, gw

    def forward(self, images):
        """images: サイズがばらばらの画像のリスト。
        返り値: feats (B, Nmax, dim)、pad_mask (B, Nmax)（Trueが詰め物）、grids [(gh, gw), ...]"""
        tokens, grids = [], []
        for img in images:
            x, gh, gw = self.patchify(img)
            x = self.patch_embed(x)
            # 基準グリッドの位置埋め込みを、この画像のパッチ数 gh×gw に補間する
            pos = F.interpolate(self.pos, size=(gh, gw), mode="bilinear", align_corners=False)
            x = x + pos[0].permute(1, 2, 0).reshape(gh * gw, -1)
            tokens.append(x)
            grids.append((gh, gw))
        # 長さの違うパッチ列を、詰め物（パディング）で同じ長さにそろえてまとめて計算する
        n_max = max(t.shape[0] for t in tokens)
        feats = torch.zeros(len(tokens), n_max, self.cfg.dim)
        pad_mask = torch.ones(len(tokens), n_max, dtype=torch.bool)
        for i, t in enumerate(tokens):
            feats[i, : t.shape[0]] = t
            pad_mask[i, : t.shape[0]] = False
        for blk in self.blocks:
            feats = blk(feats, pad_mask)
        return self.ln_post(feats), pad_mask, grids


# ---------------------------------------------------------------------------
# [0] SigLIP の事前学習（このソースだけで行う。公式は学習済みSigLIP2を使う）
# ---------------------------------------------------------------------------
class TextEncoder(nn.Module):
    """SigLIPのテキスト側。文を1本のベクトルにする（最後のトークンの位置の出力を使う）。"""

    def __init__(self, cfg, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, cfg.dim)
        self.pos = nn.Parameter(torch.randn(1, cfg.max_len, cfg.dim) * 0.02)
        self.blocks = nn.ModuleList([ViTBlock(cfg.dim, cfg.heads) for _ in range(cfg.layers)])
        self.ln = nn.LayerNorm(cfg.dim)

    def forward(self, ids, pad_mask):
        x = self.emb(ids) + self.pos[:, : ids.shape[1]]
        for blk in self.blocks:
            x = blk(x, pad_mask)
        last = (~pad_mask).sum(1) - 1
        return self.ln(x[torch.arange(len(ids)), last])


class SigLIP(nn.Module):
    """画像ベクトルと文ベクトルの内積で「この画像とこの文は対応しているか」を当てる。

    CLIPとの違い：
      - CLIP：行ごとにソフトマックス（バッチ内の全ペアの中から正解の1つを選ぶ）
      - SigLIP：ペアごとにシグモイド（各ペアを「対応している/いない」の2択として独立に判定）
        → バッチ全体の正規化がいらないので、大きなバッチでも計算が分けやすい
    SigLIP2は、これにキャプション生成やマスク予測などの損失を足しているが、このソースでは入れていない。
    """

    def __init__(self, vision, text, dim):
        super().__init__()
        self.vision, self.text = vision, text
        self.img_proj = nn.Linear(dim, dim, bias=False)
        self.txt_proj = nn.Linear(dim, dim, bias=False)
        self.t = nn.Parameter(torch.tensor(math.log(10.0)))   # 温度（論文の初期値 log10）
        self.b = nn.Parameter(torch.tensor(-10.0))            # バイアス（論文の初期値 -10）

    def embed_image(self, images):
        feats, pad, _ = self.vision(images)
        keep = (~pad).unsqueeze(-1).float()
        # 公式SigLIPはアテンションプーリング（MAP head）で1本にまとめる。ここでは簡略化して平均を使う。
        pooled = (feats * keep).sum(1) / keep.sum(1)
        return F.normalize(self.img_proj(pooled), dim=-1)

    def loss(self, images, ids, pad_mask):
        zi = self.embed_image(images)                                  # (B, D)
        zt = F.normalize(self.txt_proj(self.text(ids, pad_mask)), dim=-1)
        logits = zi @ zt.T * self.t.exp() + self.b                     # (B, B) 全ペアの点数
        labels = 2 * torch.eye(len(zi)) - 1                            # 対角（正しいペア）=+1、それ以外=-1
        return -F.logsigmoid(labels * logits).sum() / len(zi)


# ---------------------------------------------------------------------------
# [2] コネクタ（PixelUnshuffle → 2層MLP）
# ---------------------------------------------------------------------------
class Connector(nn.Module):
    """ビジョンエンコーダの出力を、言語モデルの入力（画像トークン）に変える。

    1. PixelUnshuffle：隣り合う 2×2 パッチのベクトルを横につなげて1本にする。
       トークン数は1/4、1本の長さは4倍。情報は捨てずに、並べ方だけ変える。
       → 言語モデルに入るトークン数が減るので、そのぶん計算が軽くなる（言語モデルの計算は長さに比例して重くなる）。
    2. 2層MLP：つなげたベクトルを、言語モデルの次元に変換する。
       公式：LayerNorm → Linear → GELU → Linear（LFM2-VLの公開コードの構成。LayerNormの有無は版によって違う可能性あり）
    """

    def __init__(self, cfg, vis_dim, lm_dim):
        super().__init__()
        self.f = cfg.unshuffle
        in_dim = vis_dim * self.f * self.f
        self.mlp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, lm_dim))

    def pixel_unshuffle(self, x, gh, gw):
        """x: (gh*gw, D) → ((gh/f)*(gw/f), D*f*f)"""
        f = self.f
        d = x.shape[-1]
        x = x.reshape(gh // f, f, gw // f, f, d).permute(0, 2, 1, 3, 4)
        return x.reshape((gh // f) * (gw // f), f * f * d)

    def forward(self, feats, grids):
        out = []
        for i, (gh, gw) in enumerate(grids):
            x = feats[i, : gh * gw]                     # 詰め物を外す
            out.append(self.mlp(self.pixel_unshuffle(x, gh, gw)))
        return out                                       # 画像ごとに長さの違うリスト


# ---------------------------------------------------------------------------
# [3] 言語モデル（LFM2：ゲート付き短い畳み込み ＋ GQA）
# ---------------------------------------------------------------------------
class ShortConv(nn.Module):
    """ゲート付き短い畳み込み（LFM2の主役の演算子）。

      B, C, h~ = Linear(x) を3つに分ける
      y = B ⊙ h~           （入力側のゲート）
      z = 深さ方向1次元畳み込み(y)、カーネル k=3、系列方向、因果的（過去の2トークン＋自分だけを見る）
      out = Linear(C ⊙ z)  （出力側のゲート）

    アテンションと違って、見る範囲は直前の数トークンだけ。そのかわり計算量は長さに比例するだけで、
    推論時に覚えておくものも「直前 k-1 トークン分」だけなので、CPUやスマホで速い。
    遠くのトークンとの関係は、ところどころに入れたGQA層が受け持つ。
    """

    def __init__(self, dim, k):
        super().__init__()
        self.k = k
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.conv = nn.Conv1d(dim, dim, k, groups=dim, bias=False)   # groups=dim で深さ方向（チャネルごと）
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, c, h = self.in_proj(x).chunk(3, dim=-1)
        y = (b * h).transpose(1, 2)                          # (B, D, T)
        y = F.pad(y, (self.k - 1, 0))                        # 左だけ詰め物 → 未来を見ない（因果的）
        z = self.conv(y).transpose(1, 2)
        return self.out_proj(c * z)


class GQA(nn.Module):
    """グループ化クエリアテンション。クエリのヘッドはたくさん、キー・バリューのヘッドは少なく持ち、
    複数のクエリヘッドで同じキー・バリューを共有する。推論時に保存するKVキャッシュが小さくなる。
    公式LFM2はRoPEとQK-Norm（QとKをRMSNormしてから内積）を使う。因果マスクつき。"""

    def __init__(self, dim, heads, kv_heads):
        super().__init__()
        self.h, self.kv, self.d = heads, kv_heads, dim // heads
        self.q = nn.Linear(dim, heads * self.d, bias=False)
        self.k = nn.Linear(dim, kv_heads * self.d, bias=False)
        self.v = nn.Linear(dim, kv_heads * self.d, bias=False)
        self.o = nn.Linear(heads * self.d, dim, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.d), RMSNorm(self.d)

    def forward(self, x):
        bsz, t, _ = x.shape
        q = self.q_norm(self.q(x).view(bsz, t, self.h, self.d)).transpose(1, 2)
        k = self.k_norm(self.k(x).view(bsz, t, self.kv, self.d)).transpose(1, 2)
        v = self.v(x).view(bsz, t, self.kv, self.d).transpose(1, 2)
        pos = torch.arange(t, device=x.device)
        q, k = rope(q, pos), rope(k, pos)
        rep = self.h // self.kv                               # 1つのKVヘッドを何個のQヘッドで使うか
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(bsz, t, -1))


class LMBlock(nn.Module):
    """前にRMSNorm（pre-norm）→ 演算子（畳み込み or GQA）→ 残差 → RMSNorm → SwiGLU → 残差。"""

    def __init__(self, kind, cfg):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        self.op = ShortConv(cfg.dim, cfg.conv_kernel) if kind == "c" else GQA(cfg.dim, cfg.heads, cfg.kv_heads)
        self.ff = SwiGLU(cfg.dim, cfg.ff_hidden)

    def forward(self, x):
        x = x + self.op(self.n1(x))
        return x + self.ff(self.n2(x))


class LFM2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = nn.ModuleList([LMBlock(k, cfg) for k in cfg.layout])
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab, bias=False)
        self.head.weight = self.emb.weight                     # 入力と出力で埋め込みを共有（公式も共有）

    def forward_embeds(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))                          # (B, T, vocab)


# ---------------------------------------------------------------------------
# 全体：LFM2.5-VL
# ---------------------------------------------------------------------------
class LFMVL(nn.Module):
    def __init__(self, cfg, image_token_id):
        super().__init__()
        self.vision = VisionEncoder(cfg.vision)
        self.connector = Connector(cfg.connector, cfg.vision.dim, cfg.lm.dim)
        self.lm = LFM2(cfg.lm)
        self.image_token_id = image_token_id

    def build_inputs(self, ids, images):
        """文のトークンを埋め込みに変え、<image> の位置を画像トークンで置き換える。
        画像トークンも文のトークンも、ここから先は同じ1本の列として言語モデルに入る。"""
        x = self.lm.emb(ids)                                     # (B, T, D)
        feats, _, grids = self.vision(images)
        img_tokens = self.connector(feats, grids)
        x = x.clone()
        for i, tok in enumerate(img_tokens):
            slots = (ids[i] == self.image_token_id).nonzero().squeeze(-1)
            assert len(slots) == len(tok), "<image> の数と画像トークンの数が合わない"
            x[i, slots] = tok
        return x

    def forward(self, ids, images, labels=None):
        logits = self.lm.forward_embeds(self.build_inputs(ids, images))
        if labels is None:
            return logits
        # 次のトークン予測：位置tの出力で、位置t+1のトークンを当てる
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, images, eos_id, max_new=12):
        """1トークンずつ足していく（貪欲法：一番確率の高いトークンを選ぶ）。
        公式の推論はKVキャッシュと畳み込みの状態を持ち回って速くするが、ここでは毎回全体を計算し直す。"""
        out = []
        for _ in range(max_new):
            nxt = self.forward(ids, images)[:, -1].argmax(-1)
            if nxt.item() == eos_id:
                break
            out.append(nxt.item())
            ids = torch.cat([ids, nxt[:, None]], dim=1)
        return out


def n_image_tokens(h, w, cfg):
    """画像サイズから、言語モデルに入る画像トークンの数を計算する。
    公式例：512×512、パッチ16 → 32×32=1024パッチ → PixelUnshuffle(2×2) → 256トークン"""
    p, f = cfg.vision.patch, cfg.connector.unshuffle
    return (h // p // f) * (w // p // f)
