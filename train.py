"""学習の段階（公式LFM2-VLの段階に合わせている）。

  stage0  SigLIPの事前学習（このソースだけ。公式は学習済みSigLIP2を使う）
  stage1  コネクタだけ学習（エンコーダと言語モデルは止める）    ← 公式 Stage1：connector pre-training
  stage2  全部を学習（学習率の比 言語モデル:コネクタ:エンコーダ） ← 公式 Stage2/3：mid-training と SFT

公式は Stage2 の前に、言語モデル自体をテキストだけで大量に学習してある（LFM2.5の事前学習）。
このソースでは、言語モデルは stage1 から画像の説明文で学ぶだけにしている。

使い方: python train.py   （CPUで数分〜十数分程度を想定）
"""
import random

import torch

from config import Config
from data import PAD, VOCAB, IMG, chat_tokens, encode, sample
from model import LFMVL, SigLIP, TextEncoder, n_image_tokens

torch.manual_seed(0)
random.seed(0)

cfg = Config()
cfg.lm.vocab = len(VOCAB)
PATCH_MULT = cfg.vision.patch * cfg.connector.unshuffle      # 辺の長さはこの倍数にする

model = LFMVL(cfg, image_token_id=IMG)


def pad_batch(seqs, value):
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (n - len(s)) for s in seqs])


# ---------------------------------------------------------------------------
def stage0_siglip(steps=300, bsz=16):
    # 図7-B（このソースだけの段階）
    text = TextEncoder(cfg.text_enc, len(VOCAB))
    siglip = SigLIP(model.vision, text, cfg.vision.dim)
    opt = torch.optim.AdamW(siglip.parameters(), lr=3e-4)
    for step in range(steps):
        imgs, caps = zip(*[sample(PATCH_MULT) for _ in range(bsz)])
        ids = [encode(c) for c in caps]
        ids_t = pad_batch(ids, PAD)
        loss = siglip.loss(list(imgs), ids_t, ids_t == PAD)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"[stage0 SigLIP] step {step} loss {loss.item():.3f}")


def vlm_batch(bsz):
    imgs, ids, labels = [], [], []
    for _ in range(bsz):
        img, cap = sample(PATCH_MULT)
        n = n_image_tokens(img.shape[1], img.shape[2], cfg)
        i, l = chat_tokens(n, cap)
        imgs.append(img)
        ids.append(i)
        labels.append(l)
    return imgs, pad_batch(ids, PAD), pad_batch(labels, -100)


def run_vlm(name, opt, steps, bsz=8):
    for step in range(steps):
        imgs, ids, labels = vlm_batch(bsz)
        _, loss = model(ids, imgs, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"[{name}] step {step} loss {loss.item():.3f}")


def stage1_connector(steps=200):
    # 図7-C ①
    # エンコーダと言語モデルを止める（requires_grad=False で勾配を計算しない）
    for p in model.vision.parameters():
        p.requires_grad = False
    for p in model.lm.parameters():
        p.requires_grad = False
    # 公式：コネクタだけ高めの学習率
    opt = torch.optim.AdamW(model.connector.parameters(), lr=1e-3)
    run_vlm("stage1 connector", opt, steps)


def stage2_all(steps=600, base_lr=3e-4):
    # 図7-C ②
    for p in model.parameters():
        p.requires_grad = True
    # 公式 Stage2：言語モデル:コネクタ:エンコーダ = 5:5:1 の学習率の比（エンコーダは学習済みなので壊さないよう小さく）
    opt = torch.optim.AdamW([
        {"params": model.lm.parameters(), "lr": base_lr},
        {"params": model.connector.parameters(), "lr": base_lr},
        {"params": model.vision.parameters(), "lr": base_lr / 5},
    ], weight_decay=1e-6)
    run_vlm("stage2 all", opt, steps)


if __name__ == "__main__":
    stage0_siglip()
    stage1_connector()
    stage2_all()
    torch.save(model.state_dict(), "lfm_vl_small.pt")
    print("saved lfm_vl_small.pt")
