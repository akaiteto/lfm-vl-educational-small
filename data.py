"""合成データ：図形（丸・四角・三角）を描いた画像と、その説明文。

公式は数十億トークン規模の画像キャプション、OCR、VQAなどを使う。
このソースは仕組みを追うためだけなので、ランダムに図形を描いた画像と、
決まった形の文（例：「a red circle on the left」）だけを使う。
画像サイズもわざとばらばらにして、NaFlex（縦横比を保ったまま可変サイズで入れる）の動きを見られるようにしている。
"""
import random

import torch

COLORS = {"red": (0.9, 0.2, 0.2), "green": (0.2, 0.8, 0.3), "blue": (0.2, 0.3, 0.9)}
SHAPES = ["circle", "square", "triangle"]
PLACES = ["left", "right", "top", "bottom", "center"]

# 特別なトークン。公式も画像の始まり・終わりを示すトークンで画像部分を囲む。
SPECIAL = ["<pad>", "<bos>", "<eos>", "<image_start>", "<image>", "<image_end>", "<user>", "<assistant>"]
WORDS = ["a", "the", "on", "what", "is", "in", "image", "describe", "?"] + list(COLORS) + SHAPES + PLACES
VOCAB = SPECIAL + WORDS
STOI = {w: i for i, w in enumerate(VOCAB)}
PAD, BOS, EOS = STOI["<pad>"], STOI["<bos>"], STOI["<eos>"]
IMG_START, IMG, IMG_END = STOI["<image_start>"], STOI["<image>"], STOI["<image_end>"]


def encode(text):
    return [STOI[w] for w in text.split()]


def decode(ids):
    return " ".join(VOCAB[i] for i in ids)


def draw(h, w, color, shape, place):
    """h×w の画像に図形を1つ描く。値は0〜1、形は (3, h, w)。"""
    img = torch.ones(3, h, w) * 0.95
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    cy, cx = {"left": (h / 2, w / 4), "right": (h / 2, 3 * w / 4), "top": (h / 4, w / 2),
              "bottom": (3 * h / 4, w / 2), "center": (h / 2, w / 2)}[place]
    r = min(h, w) / 6
    if shape == "circle":
        m = (ys - cy) ** 2 + (xs - cx) ** 2 < r ** 2
    elif shape == "square":
        m = ((ys - cy).abs() < r) & ((xs - cx).abs() < r)
    else:
        m = (ys - cy > -r) & (ys - cy < r) & ((xs - cx).abs() < (ys - cy + r) / 2)
    for c in range(3):
        img[c][m] = COLORS[color][c]
    return img


def sample(patch_mult=16):
    """画像1枚と説明文を1つ作る。辺の長さは patch_mult（パッチ×アンシャッフル）の倍数にする。

    公式も「辺の長さが パッチサイズ×PixelUnshuffleの係数 で割り切れること」を条件にしている。
    """
    h = patch_mult * random.randint(2, 4)
    w = patch_mult * random.randint(2, 4)
    color, shape, place = random.choice(list(COLORS)), random.choice(SHAPES), random.choice(PLACES)
    img = draw(h, w, color, shape, place)
    caption = f"a {color} {shape} on the {place}"
    return img, caption


def chat_tokens(n_img_tokens, answer):
    """VLMに入れるトークン列を作る。

    形：<bos> <user> <image_start> <image>×n <image_end> describe the image <assistant> 答え <eos>
    <image> の位置には、あとで画像トークン（コネクタの出力）を差し込む。
    labels は「答えの部分だけ」次のトークンを当てさせ、それ以外は -100（損失に入れない）にする。
    """
    prompt = [BOS, STOI["<user>"], IMG_START] + [IMG] * n_img_tokens + [IMG_END] + encode("describe the image") + [STOI["<assistant>"]]
    ans = encode(answer) + [EOS]
    ids = prompt + ans
    labels = [-100] * len(prompt) + ans
    return ids, labels
