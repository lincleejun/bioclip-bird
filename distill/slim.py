"""The teacher without what identification never runs: the image tower only, stored small.

  ./run distill/slim.py models/bioclip-2.5-vith14-visual-fp16.safetensors          # every number as float16
  ./run distill/slim.py models/bioclip-2.5-vith14-visual-int8.safetensors --int8   # weight matrices as int8

The text tower only turns names into vectors, and gardener ships those (data/names/cache); it is needed again
only to embed a new name list. Try a file with `distill/serve.py --slim FILE [--half]`; load() gives the tower's
state dict back as floats. int8: each row of a weight matrix is scaled by its own largest value (KEY.q8 int8 +
KEY.s8 float16 per row); biases, norms and embeddings stay float16. The compute is still float: this shrinks
the file, not the memory in use.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

SLIM = Path(__file__).resolve().parents[1] / "models" / "bioclip-2.5-vith14-visual-int8.safetensors"
SIZE, GRID = 224, 16
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def pack(state: dict, int8: bool) -> dict:
    out = {}
    for k, v in state.items():
        if int8 and v.is_floating_point() and v.ndim >= 2 and v.shape[0] > 1 and v.numel() > 100_000:
            scale = v.flatten(1).abs().amax(1).clamp_min(1e-12) / 127
            out[k + ".q8"] = (v / scale.view(-1, *[1] * (v.ndim - 1))).round().clamp(-127, 127).to(torch.int8)
            out[k + ".s8"] = scale.half()
        else:
            out[k] = v.half() if v.is_floating_point() else v
    return out


def load(path: str) -> dict:
    return unpack(load_file(path))


def unpack(raw: dict) -> dict:
    """pack() undone: int8 rows times their scale, as float32; everything else as stored."""
    out = {}
    for k, v in raw.items():
        if k.endswith(".q8"):
            s = raw[k[:-3] + ".s8"].float()
            out[k[:-3]] = v.float() * s.view(-1, *[1] * (v.ndim - 1))
        elif not k.endswith(".s8"):
            out[k] = v
    return out


def square(im: Image.Image) -> Image.Image:
    """The teacher's framing: shortest side to 224, centre crop."""
    k = SIZE / min(im.size)
    w, h = round(im.width * k), round(im.height * k)
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    return im.convert("RGB").resize((w, h), Image.BICUBIC).crop((left, top, left + SIZE, top + SIZE))


def tower(layers: list[int]):
    """The int8 image tower computing in float16 and a function images -> {layer: (n, 256, 1280) float16 patch tokens}.
    Images must already be 224 x 224 (square())."""
    import open_clip
    from gardener.service.adapters.bioclip import snapshot_dir
    cfg = json.loads((Path(snapshot_dir()) / "open_clip_config.json").read_text())
    model = open_clip.CLIP(**cfg["model_cfg"]).visual
    model.load_state_dict(load(str(SLIM)))
    model = model.eval().half().to(DEV)
    mean, std = (torch.tensor(cfg["preprocess_cfg"][k], device=DEV).view(1, 3, 1, 1) for k in ("mean", "std"))
    got = {}
    for n in layers:
        model.transformer.resblocks[n].register_forward_hook(lambda _m, _i, o, n=n: got.__setitem__(n, o))
    assert model.transformer.batch_first

    def tokens(images: list[Image.Image]) -> dict[int, np.ndarray]:
        x = torch.from_numpy(np.stack([np.asarray(im, np.float32) / 255 for im in images])).permute(0, 3, 1, 2).to(DEV)
        with torch.no_grad():
            model(((x - mean) / std).half())
        assert all(v.shape[1] == GRID * GRID + 1 for v in got.values())
        return {n: v[:, 1:].float().cpu().numpy().astype(np.float16) for n, v in got.items()}      # token 0 is the class token
    return tokens


if __name__ == "__main__":
    from gardener.service.adapters.bioclip import BioCLIP
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    state = BioCLIP("cpu").model.visual.state_dict()
    save_file(pack(state, "--int8" in sys.argv), out)
    back = load(str(out))
    assert back.keys() == state.keys()
    worst = max(float((back[k].float() - v.float()).abs().max() / v.float().abs().max().clamp_min(1e-12)) for k, v in state.items() if v.is_floating_point())
    print(f"{out}: {out.stat().st_size / 1e9:.2f} GB, largest rounding error {worst:.2%} of a tensor's largest value")
