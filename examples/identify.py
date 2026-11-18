"""Name the bird in a cropped photo with the int8 BioCLIP 2.5 image tower and the AviList name matrix.

  pip install torch torchvision open_clip_torch safetensors numpy pillow
  python examples/identify.py crop.jpg [more.jpg ...] \
      --model bioclip-2.5-vith14-visual-int8.safetensors --names bioclip-2.5-vith14-avilist-2026-names.npz \
      [--parts parts.safetensors] [--flight flight.safetensors]      # optional heads, from bioclip-bird-heads

Input: a photo already cropped around one bird (a detector's box with some margin). Output per image: the five
most likely species by image alone (softmax over 11,172 AviList species at the model's logit scale; no location
prior), and with the optional heads: beak and eye points in the input image's pixels and P(flying).
Needs only the files above: no gardener code, no original BioCLIP checkpoint, no network.
"""
from __future__ import annotations

import argparse

import numpy as np
import open_clip
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

# open_clip_config.json of imageomics/bioclip-2.5-vith14 (revision 6e3d04e3d652): the image half only
VISION = {"image_size": 224, "layers": 32, "width": 1280, "head_width": 80, "patch_size": 14}
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
SIZE, PATCH, GRID, LAYER = 224, 14, 16, 15      # the heads read block 15's 16 x 16 patch tokens


def unpack(raw: dict) -> dict:
    """int8 file -> float state dict: KEY.q8 (int8) times KEY.s8 (one float16 scale per row); the rest as stored."""
    out = {}
    for k, v in raw.items():
        if k.endswith(".q8"):
            out[k[:-3]] = v.float() * raw[k[:-3] + ".s8"].float().view(-1, *[1] * (v.ndim - 1))
        elif not k.endswith(".s8"):
            out[k] = v.float()
    return out


def parts(h: dict, tokens: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Beak / left eye / right eye: ((3, 2) points in the 224 frame, (3,) P(visible))."""
    x = torch.nn.functional.layer_norm(tokens, (tokens.shape[-1],), h["norm.weight"], h["norm.bias"])
    x = torch.nn.functional.gelu(x @ h["fc.weight"].T + h["fc.bias"])
    o = (x @ h["out.weight"].T + h["out.bias"]).view(GRID * GRID, 3, 3)
    cell = o[..., 0].argmax(0)                                            # most likely patch per part
    off = torch.sigmoid(o[cell, torch.arange(3), 1:])                     # where inside that patch
    pts = (torch.stack([cell % GRID, cell // GRID], -1) + off) * PATCH
    vis = torch.sigmoid(x.mean(0) @ h["vis.weight"].T + h["vis.bias"])
    return pts.numpy(), vis.numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--parts")
    ap.add_argument("--flight")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    tower = open_clip.model._build_vision_tower(1024, open_clip.model.CLIPVisionCfg(**VISION))
    tower.load_state_dict(unpack(load_file(a.model)))
    tower = tower.eval().to(dev)
    got = {}
    tower.transformer.resblocks[LAYER].register_forward_hook(lambda _m, _i, o: got.__setitem__("t", o))
    pre = transforms.Compose([transforms.Resize(SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
                              transforms.CenterCrop(SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    names = np.load(a.names)
    matrix = torch.from_numpy(names["matrix"].astype(np.float32)).to(dev)
    scale = float(names["logit_scale"])
    ph = {k: v.float() for k, v in load_file(a.parts).items()} if a.parts else None
    fh = {k: v.float() for k, v in load_file(a.flight).items()} if a.flight else None

    for path in a.images:
        im = Image.open(path).convert("RGB")
        with torch.no_grad():
            f = tower(pre(im)[None].to(dev))
            f = f / f.norm(dim=-1, keepdim=True)
            p = (scale * f @ matrix.T).softmax(-1)[0].cpu()
        top = p.topk(5)
        print(path)
        for s, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"  {s:.3f}  {names['scientific'][i]} ({names['common'][i]})")
        tokens = got["t"][0, 1:].float().cpu()                              # (256, 1280); token 0 is the class token
        if ph:
            pts, vis = parts(ph, tokens)
            k = SIZE / min(im.size)                                          # 224 frame -> input pixels (undo resize + centre crop)
            left, top_ = (int(im.width * k) - SIZE) // 2, (int(im.height * k) - SIZE) // 2
            for name, (x, y), v in zip(("beak", "left eye", "right eye"), pts, vis):
                print(f"  {name:9s} ({(x + left) / k:.0f}, {(y + top_) / k:.0f})  P(visible) {v:.2f}")
        if fh:
            z = (tokens.mean(0) - fh["mean"]) / fh["std"]
            print(f"  P(flying) {torch.sigmoid(z @ fh['weight'] + fh['bias']).item():.2f}")


if __name__ == "__main__":
    main()
