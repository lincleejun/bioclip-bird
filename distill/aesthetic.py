"""Can the aesthetic score come from the teacher's tokens instead of SigLIP2? A ridge head, gardener's own recipe.

  ./run distill/aesthetic.py cv [--frame squash]      # 5-fold cross-validation per block, against gardener's SigLIP2 head
  ./run distill/aesthetic.py golden [--frame squash]  # both heads on EVA's 100 held-out images
  ./run distill/aesthetic.py own FOLDER [--frame squash]   # both heads on the owner's shoot against the Lightroom keeps (test only)

Data and recipe are gardener's (data/aesthetic/README.md): EVA, mean general score (0-10) of images with 30+
filtered votes, the 100 golden images held out, gardener.aesthetic_fit with its seed and alphas. Only the features
change: the mean of block 15's patch tokens over the whole frame. --frame crop = shortest side 224 and centre crop
(the teacher's framing, loses the sides of a 3:2 photo); squash = the whole frame pressed into 224 x 224 (what
SigLIP2's processor does). `own` reads notes/culling-bench/rows.json (Lightroom rating and gardener's score per
photo) and never writes near the photos.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from distill.slim import SIZE, square, tower

ROOT = Path(__file__).resolve().parents[1]
EVA = Path.home() / ".cache" / "bioscan" / "eva"
AES = ROOT.parent / "bioscan" / "data" / "aesthetic"
LAYERS, BEST = [7, 11, 15, 19, 23, 27, 31], 15


def framed(im: Image.Image, frame: str) -> Image.Image:
    return square(im) if frame == "crop" else im.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)


def eva(frame: str) -> tuple[list[str], np.ndarray, list[str], np.ndarray, dict[str, np.ndarray]]:
    """(train ids, their scores, golden ids, their scores, {id: block-15.. features per layer})."""
    votes = defaultdict(list)
    with open(EVA / "data" / "votes_filtered.csv", newline="") as f:
        for r in csv.DictReader(f, delimiter="="):
            votes[r["image_id"]].append(float(r["score"]))
    with open(AES / "eva-golden-v1.csv", newline="") as f:
        gold = {r["image_id"]: float(r["mean"]) for r in csv.DictReader(f)}
    jpg = lambda i: EVA / "images" / "EVA_together" / f"{i}.jpg"
    ids = sorted(i for i, v in votes.items() if len(v) >= 30 and i not in gold and jpg(i).exists())
    every = ids + sorted(gold)
    cache = ROOT / "data" / f"eva-tokens-{frame}.npz"
    if not cache.exists() or len(np.load(cache)["ids"]) != len(every):
        tok, feats = tower(LAYERS), {n: [] for n in LAYERS}
        for i in range(0, len(every), 32):
            for n, v in tok([framed(Image.open(jpg(j)), frame) for j in every[i:i + 32]]).items():
                feats[n].append(v.astype(np.float32).mean(1))
        np.savez(cache, ids=np.array(every), **{str(n): np.concatenate(v) for n, v in feats.items()})
    z = np.load(cache)
    return ids, np.array([np.mean(votes[i]) for i in ids]), sorted(gold), np.array([gold[i] for i in sorted(gold)]), {n: z[str(n)] for n in LAYERS}


def fitted(frame: str):
    """The block-15 head fit on EVA's 3,970 training images with the alpha cross-validation picks."""
    from gardener import aesthetic_fit
    ids, y, _g, _gy, feats = eva(frame)
    X = feats[BEST][:len(ids)].astype(np.float64)
    return aesthetic_fit.ridge(X, y, aesthetic_fit.cross_validate(X, y, lo=0.0, hi=10.0)["best_alpha"], None, 0.0, 10.0)


def auc(score: np.ndarray, keep: np.ndarray) -> float:
    """Chance that a kept photo scores above one that was not kept (0.5 = random)."""
    from scipy.stats import rankdata
    r = rankdata(score)
    return float((r[keep].sum() - keep.sum() * (keep.sum() + 1) / 2) / (keep.sum() * (~keep).sum()))


class Frames(torch.utils.data.Dataset):
    def __init__(self, paths: list[Path], frame: str) -> None:
        self.paths, self.frame = paths, frame

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from gardener.service.decode import decode
        return np.asarray(framed(decode(str(self.paths[i])).image, self.frame)), i


def main() -> None:
    from scipy.stats import spearmanr

    from gardener import aesthetic as aes
    from gardener import aesthetic_fit
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cv", "golden", "own"])
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--frame", choices=["crop", "squash"], default="crop")
    a = ap.parse_args()
    if a.cmd == "cv":
        ids, y, _g, _gy, feats = eva(a.frame)
        print(f"{len(ids)} images, frame={a.frame}; baseline SigLIP2 frame vector: SRCC 0.792")
        for n in LAYERS:
            cv = aesthetic_fit.cross_validate(feats[n][:len(ids)].astype(np.float64), y, lo=0.0, hi=10.0)
            print(json.dumps({"block": n, "alpha": cv["best_alpha"], "srcc": cv["best"]["srcc_mean"], "srcc_sd": cv["best"]["srcc_sd"], "plcc": cv["best"]["plcc_mean"]}), flush=True)
    elif a.cmd == "golden":
        ids, _y, gold, gy, feats = eva(a.frame)
        ours = fitted(a.frame).predict(feats[BEST][len(ids):].astype(np.float64))
        from gardener.service.adapters.siglip2 import SigLIP2
        sig = SigLIP2("mps" if torch.backends.mps.is_available() else "cpu")      # gardener's own vectors for the same 100 images
        vecs = np.concatenate([sig.embed_images([Image.open(EVA / "images" / "EVA_together" / f"{j}.jpg").convert("RGB") for j in gold[i:i + 32]]) for i in range(0, len(gold), 32)])
        theirs = aesthetic_fit.head_scores(aes.load_head(AES / "eva-head-v1.json"), np.asarray(vecs, dtype=np.float64))
        print(json.dumps({"images": len(gold), "frame": a.frame, "srcc_bioclip_block15": round(float(spearmanr(ours, gy)[0]), 4),
                          "srcc_siglip2": round(float(spearmanr(theirs, gy)[0]), 4), "srcc_between_heads": round(float(spearmanr(ours, theirs)[0]), 4)}))
    else:
        folder = Path(a.folder)
        rows = [r for r in json.loads((ROOT / "notes/culling-bench/rows.json").read_text()) if r["folder"] == folder.name and r["aes"] is not None]
        by_stem = {p.stem.lower(): p for p in folder.iterdir() if p.suffix.lower() in (".arw", ".dng", ".jpg")}
        rows = [r for r in rows if r["stem"] in by_stem]
        cache = ROOT / "data" / f"own-{folder.name}-{a.frame}.npy"
        if not cache.exists():
            tok, out = tower([BEST]), np.zeros((len(rows), 1280), np.float32)
            for x, idx in torch.utils.data.DataLoader(Frames([by_stem[r["stem"]] for r in rows], a.frame), batch_size=16, num_workers=8):
                out[idx.numpy()] = tok([Image.fromarray(v) for v in x.numpy()])[BEST].astype(np.float32).mean(1)
            np.save(cache, out)
        ours = fitted(a.frame).predict(np.load(cache).astype(np.float64)) / 10
        theirs, keep = np.array([r["aes"] for r in rows]), np.array([r["rating"] > 0 for r in rows])
        report = {"folder": folder.name, "photos": len(rows), "keeps": int(keep.sum()), "frame": a.frame, "srcc_between_heads": round(float(spearmanr(ours, theirs)[0]), 4)}
        for name, s in (("bioclip_block15", ours), ("siglip2", theirs)):
            top = np.argsort(-s)[:round(len(s) * 200 / 1500)]
            bursts = defaultdict(list)
            for i, r in enumerate(rows):
                bursts[r["burst"] or r["stem"]].append(i)
            picks = [g for g in bursts.values() if len(g) > 1 and keep[g].any()]
            report[name] = {"auc_keep": round(auc(s, keep), 4), "keeps_in_top_13pct": f"{int(keep[top].sum())}/{int(keep.sum())}",
                            "best_of_burst_is_a_keep": f"{sum(bool(keep[g[int(np.argmax(s[g]))]]) for g in picks)}/{len(picks)}",
                            "best_of_burst_by_chance": round(float(np.mean([keep[g].mean() for g in picks])), 4)}
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
