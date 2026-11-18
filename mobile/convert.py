"""Core ML files for the iOS app, written to mobile/build/ (not in git).

  ./mobile/ct mobile/convert.py bioclip [--fp16]   # BioCLIP.mlpackage: image -> embedding + block-15 tokens, 8-bit weights
  ./mobile/ct mobile/convert.py heads                                # Heads.mlpackage: tokens -> eye/beak + flight (non-commercial weights)
  ./mobile/ct mobile/convert.py detector [dfine-nano|rtdetr-r18]     # Detector.mlpackage: 640x640 image -> 300 bird scores + boxes
  ./mobile/ct mobile/convert.py names                                # names.f16 (11172 x 1024 float16) + names.tsv

BioCLIP is built from its open_clip config and the float16 slim file (distill/slim.py), traced in float32, converted
to an ML program computing in float16, then its weights are quantized to 8 bits (linear, per output channel).
The input is a 224x224 RGB image already framed the way gardener frames it (crop_with_context, shortest side to 224
with PIL bicubic, centre crop); the model only divides by 255 and applies BioCLIP's mean/std.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mobile" / "build"
NAMES = ROOT.parent / "bioscan" / "data" / "names"
LAYER = 15


def bioclip_tower():
    import open_clip

    from distill import slim
    from gardener.service.adapters.bioclip import snapshot_dir
    cfg = json.loads((Path(snapshot_dir()) / "open_clip_config.json").read_text())
    visual = open_clip.CLIP(**cfg["model_cfg"]).visual
    visual.load_state_dict({k: v.float() for k, v in slim.load(str(ROOT / "models/bioclip-2.5-vith14-visual-fp16.safetensors")).items()})
    return visual.eval(), cfg["preprocess_cfg"]


class Tower(torch.nn.Module):
    """x: (1, 3, 224, 224) in 0..255 -> (unit 1024-d embedding, (1, 256, 1280) block-15 patch tokens)."""

    def __init__(self, visual, pre) -> None:
        super().__init__()
        self.v = visual
        self.register_buffer("mean", torch.tensor(pre["mean"]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(pre["std"]).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        x = self.v._embeds((x - self.mean) / self.std)
        tokens = None
        for i, block in enumerate(self.v.transformer.resblocks):
            x = block(x)
            if i == LAYER:
                tokens = x[:, 1:]
        pooled, _ = self.v._pool(x)
        e = pooled @ self.v.proj
        return e / e.norm(dim=-1, keepdim=True), tokens


class Heads(torch.nn.Module):
    """block-15 tokens -> (heat (1,256,3), offset (1,256,3,2) in 0..1 of a patch, visible logits (1,3), flying p (1,))
    for beak, left eye, right eye. Picking the cell (argmax) is left to the app."""

    def __init__(self) -> None:
        super().__init__()
        from safetensors.torch import load_file
        nn = torch.nn                                    # the parts head as heads-repo/parts.py's Head has it
        self.norm, self.fc, self.out, self.vis = nn.LayerNorm(1280), nn.Linear(1280, 256), nn.Linear(256, 9), nn.Linear(256, 3)
        self.load_state_dict(load_file(ROOT / "runs/parts-1/parts.safetensors"))
        f = load_file(ROOT / "runs/flight-1/flight.safetensors")
        for k in ("mean", "std", "weight", "bias"):
            self.register_buffer("f_" + k, f[k])

    def forward(self, t):
        t = t.float()
        h = torch.nn.functional.gelu(self.fc(self.norm(t)))
        o = self.out(h).view(1, 256, 3, 3)
        fly = torch.sigmoid(((t.mean(1) - self.f_mean) / self.f_std * self.f_weight).sum(-1) + self.f_bias)
        return o[..., 0], torch.sigmoid(o[..., 1:]), self.vis(h.mean(1)), fly


class Detector(torch.nn.Module):
    """(1, 3, 640, 640) in 0..1 (the frame squashed to a square) -> (300,) bird score, (300, 4) x0 y0 x1 y1 in 0..1."""

    def __init__(self, model) -> None:
        super().__init__()
        self.m, self.bird = model, model.config.label2id["bird"]
        inner = model.model                              # anchors depend only on the fixed 640 input: freeze them as constants
        shapes = tuple((640 // s, 640 // s) for s in model.config.feat_strides)
        fixed = inner.generate_anchors(shapes)
        inner.generate_anchors = lambda *a, **k: fixed
        from transformers.models.d_fine import modeling_d_fine as d

        def integral(self, corners, project):        # F.linear with a 1-d weight does not convert; same sum, written out
            n, q, _ = corners.shape
            p = corners.reshape(-1, self.max_num_bins + 1).softmax(1)
            return (p * project.reshape(1, -1)).sum(1).reshape(n, q, -1)
        d.DFineIntegral.forward = integral

    def forward(self, x):
        out = self.m(pixel_values=x)
        cx, cy, w, h = out.pred_boxes[0].unbind(-1)
        return torch.sigmoid(out.logits[0, :, self.bird]), torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def save(ml, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ml.save(str(OUT / name))
    size = sum(f.stat().st_size for f in (OUT / name).rglob("*") if f.is_file())
    print(f"{OUT / name}: {size / 1e6:.1f} MB", flush=True)


def main() -> None:
    import coremltools as ct
    from coremltools.converters.mil.frontend.torch import ops
    cast = ops._cast

    def cast_item(context, node, dtype, name):       # coremltools 9.0 + torch 2.14: int() of a shape-(1,) constant fails
        x = context[node.inputs[0]]
        if x.can_be_folded_to_const() and getattr(x.val, "shape", ()) and not isinstance(x.val, dtype):
            context.add(ct.converters.mil.Builder.const(val=dtype(np.asarray(x.val).item()), name=node.name), node.name)
        else:
            cast(context, node, dtype, name)
    ops._cast = cast_item

    @ops.register_torch_op(override=True)
    def outer(context, node):                        # D-FINE's position embedding: arange(dtype=float) arrives as int32
        mb = ct.converters.mil.Builder
        x, y = (mb.cast(x=mb.reshape(x=v, shape=s), dtype="fp32") for v, s in zip(ops._get_inputs(context, node, expected=2), ([-1, 1], [1, -1])))
        context.add(mb.matmul(x=x, y=y, name=node.name))
    what = sys.argv[1]
    torch.backends.mha.set_fastpath_enabled(False)
    if what == "bioclip":
        tower = Tower(*bioclip_tower()).eval()
        with torch.no_grad():
            traced = torch.jit.trace(tower, torch.rand(1, 3, 224, 224) * 255)
        ml = ct.convert(traced, inputs=[ct.ImageType(name="image", shape=(1, 3, 224, 224), color_layout=ct.colorlayout.RGB)],
                        outputs=[ct.TensorType(name="embedding"), ct.TensorType(name="tokens15", dtype=np.float16)],
                        minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT16)
        if "--fp16" in sys.argv:
            return save(ml, "BioCLIP-fp16.mlpackage")
        from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights
        ml = linear_quantize_weights(ml, OptimizationConfig(OpLinearQuantizerConfig(mode="linear_symmetric", granularity="per_channel")))
        save(ml, "BioCLIP.mlpackage")
    elif what == "heads":
        with torch.no_grad():
            traced = torch.jit.trace(Heads().eval(), torch.randn(1, 256, 1280))
        ml = ct.convert(traced, inputs=[ct.TensorType(name="tokens15", shape=(1, 256, 1280), dtype=np.float16)],
                        outputs=[ct.TensorType(name=n) for n in ("heat", "offset", "visible", "flying")],
                        minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT32)
        save(ml, "Heads.mlpackage")
    elif what == "detector":
        from transformers import AutoModelForObjectDetection

        from distill.detectors import HF
        name = sys.argv[2] if len(sys.argv) > 2 else "dfine-nano"
        det = Detector(AutoModelForObjectDetection.from_pretrained(HF[name]).eval()).eval()
        with torch.no_grad():
            traced = torch.jit.trace(det, torch.rand(1, 3, 640, 640), strict=False)
        ml = ct.convert(traced, inputs=[ct.ImageType(name="image", shape=(1, 3, 640, 640), scale=1 / 255, color_layout=ct.colorlayout.RGB)],
                        outputs=[ct.TensorType(name="scores"), ct.TensorType(name="boxes")],
                        minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT32)
        ml.short_description = f"{HF[name]} (Apache-2.0), bird class only"
        save(ml, "Detector.mlpackage")
    elif what == "names":
        z = np.load(NAMES / "cache/bioclip-2.5-vith14-a4f7858b14c15128.npz")
        OUT.mkdir(parents=True, exist_ok=True)
        z["matrix"].astype("<f2").tofile(OUT / "names.f16")
        import csv
        with open(NAMES / "names-zh.csv", newline="") as f:
            zh = {r["scientific"]: r["zh"] for r in csv.DictReader(f)}
        with open(OUT / "names.tsv", "w") as f:       # scientific, English, Chinese, genus, family; row i = matrix row i
            for s, c, t in zip(z["scientific"], z["common"], z["taxonomy"]):
                f.write("\t".join([s, c, zh.get(s, ""), t[5], t[4]]) + "\n")
        print(f"names: {len(z['scientific'])} rows, {sum(bool(zh.get(s)) for s in z['scientific'])} with a Chinese name")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
