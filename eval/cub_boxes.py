"""Box accuracy against true boxes: each detector on the CUB-200-2011 test split (5,794 photos, one bird each, one box).

  ./run eval/cub_boxes.py NAME [NAME ...]     # NAME: owlv2, owlv2-int8, or a distill/detectors.py name; needs data/cub
  -> eval/results/boxes-cub-NAME/boxes.jsonl.gz (per photo) and a summary line on stdout

Thresholds as in the pipeline: OWLv2 keeps "a bird" boxes scoring 0.3 or more, a COCO detector its bird boxes at
0.25 or more; a photo with none of those takes what reaches 0.1 (OWLv2) or 0.125 (COCO), as the engine's second
ask does. The answer is the highest-scoring box. CUB photos and boxes are not redistributed.
"""
from __future__ import annotations

import gzip
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CUB = ROOT / "data" / "cub" / "CUB_200_2011"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def test_split() -> list[tuple[str, tuple[float, float, float, float]]]:
    rows = lambda n: [line.split() for line in (CUB / n).read_text().splitlines()]   # noqa: E731
    path = {i: p for i, p in rows("images.txt")}
    box = {i: (float(x), float(y), float(x) + float(w), float(y) + float(h)) for i, x, y, w, h in rows("bounding_boxes.txt")}
    return [(path[i], box[i]) for i, t in rows("train_test_split.txt") if t == "0"]


def detector(name: str):
    """image -> (primary threshold, [(score, box in pixels)] down to the second-ask floor)."""
    if name.startswith("owlv2"):
        from gardener.service.adapters.owlv2 import OWLv2

        from distill.detectors import owlv2_int8
        det = owlv2_int8(DEV) if name == "owlv2-int8" else OWLv2(DEV)
        return 0.3, lambda im: [(d.confidence, d.bbox) for d in det.detect(im, ["a bird"], threshold=0.1)]
    from distill.detectors import HF, Coco
    det = Coco(name, DEV if name in HF else "cpu", 0.25)
    return 0.25, det.boxes


def iou(a, b) -> float:
    w, h = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def run(name: str) -> None:
    t, boxes = detector(name)
    out = ROOT / "eval" / "results" / f"boxes-cub-{name}"
    out.mkdir(parents=True, exist_ok=True)
    rows, ms = [], []
    with gzip.open(out / "boxes.jsonl.gz", "wt") as f:
        for path, truth in test_split():
            im = Image.open(CUB / "images" / path).convert("RGB")
            s = time.perf_counter()
            found = boxes(im)
            ms.append(1000 * (time.perf_counter() - s))
            kept = [b for b in found if b[0] >= t] or found
            best = max(kept, key=lambda b: b[0]) if kept else None
            r = {"image": path, "boxes": len(kept),                 # CUB's own boxes are not republished
                 "best": [round(v, 1) for v in best[1]] if best else None, "score": round(best[0], 4) if best else None,
                 "iou": round(iou(best[1], truth), 4) if best else 0.0}
            rows.append(r)
            f.write(json.dumps(r) + "\n")
    n, ious = len(rows), sorted(r["iou"] for r in rows if r["best"])
    summary = {"detector": name, "photos": n, "device": DEV if not name.startswith("yolo") else "cpu",
               "found": round(sum(r["best"] is not None for r in rows) / n, 4),
               "iou_mean": round(statistics.mean(ious), 4), "iou_median": round(statistics.median(ious), 4),
               "iou50": round(sum(r["iou"] >= 0.5 for r in rows) / n, 4), "iou75": round(sum(r["iou"] >= 0.75 for r in rows) / n, 4),
               "boxes_per_photo": round(statistics.mean(r["boxes"] for r in rows), 3),
               "ms_median_while_benchmarking": round(statistics.median(ms), 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        run(name)
