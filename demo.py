"""学習した重みで、画像の説明文を生成してみる。

使い方: python demo.py   （先に python train.py を実行して lfm_vl_small.pt を作っておく）
"""
import random

import torch

from config import Config
from data import BOS, EOS, IMG, IMG_END, IMG_START, STOI, VOCAB, decode, encode, sample
from model import LFMVL, n_image_tokens

cfg = Config()
cfg.lm.vocab = len(VOCAB)
model = LFMVL(cfg, image_token_id=IMG)
model.load_state_dict(torch.load("lfm_vl_small.pt"))
model.eval()

random.seed(1)
for _ in range(5):
    img, cap = sample(cfg.vision.patch * cfg.connector.unshuffle)
    h, w = img.shape[1], img.shape[2]
    n = n_image_tokens(h, w, cfg)
    prompt = [BOS, STOI["<user>"], IMG_START] + [IMG] * n + [IMG_END] + encode("describe the image") + [STOI["<assistant>"]]
    out = model.generate(torch.tensor([prompt]), [img], eos_id=EOS)
    print(f"画像 {h}x{w}（画像トークン {n}個）  正解: {cap}  /  生成: {decode(out)}")
