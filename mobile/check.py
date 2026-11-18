"""Does the Core ML conversion agree with PyTorch? And what should the app answer for a given photo?

  ./mobile/ct mobile/check.py golden [--n 600]     # BioCLIP + heads on recorded crops (data/rec/golden): Core ML vs PyTorch
  ./mobile/ct mobile/check.py detector [--n 60]    # Core ML D-FINE vs transformers on eval photos: same best bird box?
  ./mobile/ct mobile/check.py expect IMG...        # the app's whole path in Python -> mobile/expected/ (read by BirdIDTests)

The app's path (identify): upright photo -> long edge 2048 (and a <= 3072 copy for crops), as gardener's decode does ->
the 2048 frame squashed to 640x640 (PIL bilinear) -> D-FINE, bird boxes scoring >= 0.25 (>= 0.125 if none) -> the
largest box -> gardener's species_crops -> open_clip's resize (shortest side 224, bicubic) + centre crop -> BioCLIP ->
98.86 x cosine against the 11,172 AviList names, softmax. No location prior, no SigLIP gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "mobile" / "build"
EXPECTED = ROOT / "mobile" / "expected"
NAMES = ROOT.parent / "bioscan" / "data" / "names" / "cache"
SCALE = 98.8644790649414                                   # data/rec/golden/logit_scale.txt
T, GRID, PATCH = 0.25, 16, 14


def mlmodel(name: str, units: str = "ALL"):
    import coremltools as ct
    return ct.models.MLModel(str(BUILD / name), compute_units=getattr(ct.ComputeUnit, units))


def frame224(crop: Image.Image) -> Image.Image:
    import open_clip
    t = open_clip.image_transform(224, is_train=False, resize_mode="shortest", interpolation="bicubic")
    return t.transforms[1](t.transforms[0](crop.convert("RGB")))


def names() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(NAMES / "bioclip-2.5-vith14-a4f7858b14c15128.npz")
    return z["matrix"].astype(np.float16).astype(np.float32), z["scientific"]   # the app reads the float16 copy


def points(heat: np.ndarray, off: np.ndarray, vis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(3, 2) beak / left eye / right eye in the 224 px frame, (3,) visible probability."""
    cell = heat.reshape(GRID * GRID, 3).argmax(0)
    o = off.reshape(GRID * GRID, 3, 2)[cell, np.arange(3)]
    return (np.stack([cell % GRID, cell // GRID], -1) + o) * PATCH, 1 / (1 + np.exp(-vis.reshape(3)))


def golden(n: int) -> None:
    import coremltools as ct
    sys.path.insert(0, str(ROOT / "mobile"))
    from convert import Heads, Tower, bioclip_tower
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tower, heads = Tower(*bioclip_tower()).eval().to(dev), Heads().eval()
    bird_m, _ = names()
    mammal_m = np.load(NAMES / "bioclip-2.5-vith14-3eab033f008ac62f.npz")["matrix"].astype(np.float32)
    rec = ROOT / "data/rec/golden"
    crops = []                                             # (jpg, recorded teacher vector) for crops the teacher calls a bird
    for npy in sorted(rec.glob("*.npy")):
        v = np.load(npy).astype(np.float32)
        for i, f in enumerate(v):
            if (f @ bird_m.T).max() > (f @ mammal_m.T).max():
                crops.append((rec / f"{npy.stem}_{i:02d}.jpg", f))
    print(f"{len(crops)} bird crops of {sum(1 for _ in rec.glob('*.jpg'))}; using {min(n, len(crops))}", flush=True)
    crops = crops[:: max(1, len(crops) // n)][:n]
    ml, mh = mlmodel("BioCLIP.mlpackage"), mlmodel("Heads.mlpackage")
    ml_cpu = mlmodel("BioCLIP.mlpackage", "CPU_ONLY")
    rows = []
    for k, (jpg, teacher) in enumerate(crops):
        im = frame224(Image.open(jpg))
        x = torch.from_numpy(np.asarray(im, np.float32)).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad():
            e_pt, tok_pt = (t.float().cpu() for t in tower(x))
            h_pt = [t.numpy() for t in heads(tok_pt)]
        out = ml.predict({"image": im})
        e_ml = out["embedding"].reshape(-1)
        h_ml = mh.predict({"tokens15": out["tokens15"].astype(np.float16)})
        h_ml = [h_ml[k2] for k2 in ("heat", "offset", "visible", "flying")]
        e_pt = e_pt.numpy().reshape(-1)
        p_pt, v_pt = points(*h_pt[:3])
        p_ml, v_ml = points(*h_ml[:3])
        row = {"cos_ml_pt": float(e_ml @ e_pt), "cos_ml_teacher": float(e_ml @ teacher), "cos_pt_teacher": float(e_pt @ teacher),
               "top1_ml_pt": int((e_ml @ bird_m.T).argmax() == (e_pt @ bird_m.T).argmax()),
               "top1_ml_teacher": int((e_ml @ bird_m.T).argmax() == (teacher @ bird_m.T).argmax()),
               "top1_pt_teacher": int((e_pt @ bird_m.T).argmax() == (teacher @ bird_m.T).argmax()),
               "point_px": np.linalg.norm(p_ml - p_pt, axis=1).tolist(), "visible_same": ((v_ml > 0.5) == (v_pt > 0.5)).tolist(),
               "fly_same": int((float(h_ml[3].reshape(-1)[0]) > 0.5) == (float(h_pt[3].reshape(-1)[0]) > 0.5)),
               "fly_diff": abs(float(h_ml[3].reshape(-1)[0]) - float(h_pt[3].reshape(-1)[0]))}
        if k < 50:                                         # the simulator runs Core ML on the CPU
            e_cpu = ml_cpu.predict({"image": im})["embedding"].reshape(-1)
            row["cos_cpu_pt"], row["top1_cpu_pt"] = float(e_cpu @ e_pt), int((e_cpu @ bird_m.T).argmax() == (e_pt @ bird_m.T).argmax())
        rows.append(row)
        if k % 100 == 0:
            print(k, flush=True)
    rep = {"crops": len(rows)}
    for key in ("cos_ml_pt", "cos_ml_teacher", "cos_pt_teacher", "cos_cpu_pt"):
        v = np.array([r[key] for r in rows if key in r])
        rep[key] = {"mean": round(float(v.mean()), 5), "min": round(float(v.min()), 5), "n": len(v)}
    for key in ("top1_ml_pt", "top1_ml_teacher", "top1_pt_teacher", "top1_cpu_pt", "fly_same"):
        v = [r[key] for r in rows if key in r]
        rep[key] = f"{sum(v)}/{len(v)}"
    pts = np.array([r["point_px"] for r in rows])
    rep["point_px_beak_lefteye_righteye"] = {"median": np.median(pts, 0).round(2).tolist(), "p95": np.percentile(pts, 95, 0).round(2).tolist(),
                                             "within_3px": (pts <= 3).mean(0).round(4).tolist()}
    rep["visible_same"] = np.array([r["visible_same"] for r in rows]).mean(0).round(4).tolist()
    rep["fly_diff_max"] = round(max(r["fly_diff"] for r in rows), 4)
    print(json.dumps(rep, indent=1))


def boxes(scores: np.ndarray, xyxy: np.ndarray, w: int, h: int) -> list[tuple[float, list[float]]]:
    for t in (T, T / 2):
        keep = []                                          # best first; IoU > 0.5 with a kept box = the same bird (as the app does)
        for i in sorted(np.flatnonzero(scores >= t), key=lambda i: -scores[i]):
            b = (xyxy[i] * [w, h, w, h]).tolist()
            if all(iou(k[1], b) <= 0.5 for k in keep):
                keep.append((float(scores[i]), b))
        if keep:
            return keep
    return []


def largest(found):
    return max(found, key=lambda f: (f[1][2] - f[1][0]) * (f[1][3] - f[1][1]))


def iou(a, b) -> float:
    from gardener.service.rules import iou as g
    return g(tuple(a), tuple(b))


def detector(n: int) -> None:
    import csv

    from distill.detectors import Coco
    from gardener.service.decode import decode
    ref, det = Coco("dfine-nano", "cpu", T), mlmodel("Detector.mlpackage")
    with open(ROOT.parent / "bioscan/data/inat/groundtruth-inat.csv", newline="") as f:
        paths = [r["path"] for r in csv.DictReader(f) if r["kind"] == "bird"]
    paths = paths[:: max(1, len(paths) // n)][:n]
    same, ious, none_both = 0, [], 0
    for p in paths:
        im = decode(p).image
        a = [(s, list(b)) for s, b in ref.boxes(im) if s >= T] or [(s, list(b)) for s, b in ref.boxes(im)]
        o = det.predict({"image": im.resize((640, 640), Image.BILINEAR)})
        b = boxes(o["scores"].reshape(-1), o["boxes"].reshape(-1, 4), *im.size)
        if not a and not b:
            none_both += 1
            continue
        if a and b:
            ious.append(iou(largest(a)[1], largest(b)[1]))
    ious = np.array(ious)
    print(json.dumps({"photos": len(paths), "no_bird_in_both": none_both, "both_found": len(ious),
                      "largest_box_iou_median": round(float(np.median(ious)), 4), "iou_ge_0.9": int((ious >= 0.9).sum()),
                      "iou_lt_0.5": int((ious < 0.5).sum())}, indent=1))


def expect(paths: list[str]) -> None:
    from gardener.service.decode import decode
    from gardener.service.rules import species_crops
    det, ml, mh = mlmodel("Detector.mlpackage"), mlmodel("BioCLIP.mlpackage"), mlmodel("Heads.mlpackage")
    m, sci = names()
    EXPECTED.mkdir(exist_ok=True)
    out = []
    for p in paths:
        d = decode(p, 3072)
        o = det.predict({"image": d.image.resize((640, 640), Image.BILINEAR)})
        found = boxes(o["scores"].reshape(-1), o["boxes"].reshape(-1, 4), *d.image.size)
        if not found:
            out.append({"path": os.path.relpath(p, ROOT), "box": None})
            continue
        score, box = largest(found)
        im = frame224(species_crops(d.image, [tuple(box)], d.detail)[0])
        im.save(EXPECTED / f"{len(out)}.png")           # the 224 px input, to compare pixel by pixel
        r = ml.predict({"image": im})
        e = r["embedding"].reshape(-1)
        logit = SCALE * (e @ m.T)
        prob = np.exp(logit - logit.max())
        prob /= prob.sum()
        top = prob.argsort()[::-1][:5]
        h = mh.predict({"tokens15": r["tokens15"].astype(np.float16)})
        pts, vis = points(h["heat"], h["offset"], h["visible"])
        out.append({"path": os.path.relpath(p, ROOT), "size": list(d.image.size), "box": [round(v, 2) for v in box], "box_score": round(score, 4),
                    "embedding": e.round(5).tolist(),
                    "top5": [{"scientific": str(sci[i]), "p": round(float(prob[i]), 4)} for i in top],
                    "points224": pts.round(2).tolist(), "visible": vis.round(4).tolist(), "flying": round(float(h["flying"].reshape(-1)[0]), 4)})
    (EXPECTED / "expected.json").write_text(json.dumps(out, indent=1))
    print(json.dumps([{k: v for k, v in o.items() if k != "embedding"} for o in out], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["golden", "detector", "expect"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    {"golden": lambda: golden(a.n), "detector": lambda: detector(a.n), "expect": lambda: expect(a.paths)}[a.cmd]()


if __name__ == "__main__":
    main()
