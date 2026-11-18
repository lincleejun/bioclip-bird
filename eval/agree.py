"""Comparisons computed from the published per-photo predictions (eval/results/*/preds.ndjson[.gz]), no model run.

  python eval/agree.py species > eval/results/agree-species.md   # species models: photo-by-photo against the original
  python eval/agree.py boxes   > eval/results/agree-boxes.md     # detectors: their best box against OWLv2's

Truth labels are renamed with gardener's synonyms the way `gardener-engine eval` does it (inat + spelling rows of
data/names/synonyms.csv, read from ../bioscan when present; without it a few dozen truths stay unrenamed).
Scoring follows gardener's eval: the highest-scoring box is the answer; a photo without one is a miss.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SYN = HERE.parents[1] / "bioscan" / "data" / "names" / "synonyms.csv"


def norm(s: str) -> str:
    """gardener's naming.norm_binomial: underscores and hyphens to spaces, no apostrophes, lower case."""
    return re.sub(r"\s+", " ", (s or "").replace("_", " ").replace("-", " ").replace("'", "")).strip().lower()


def truth() -> dict[str, dict]:
    alias = {}
    if SYN.exists():
        with open(SYN, newline="") as f:
            alias = {norm(r["alias"]): r["avilist_scientific"].strip() for r in csv.DictReader(f) if r["source"].strip() in ("inat", "spelling")}
    with open(HERE / "groundtruth.csv", newline="") as f:
        return {r["path"]: {**r, "scientific": alias.get(norm(r["scientific"]), r["scientific"])} for r in csv.DictReader(f)}


def preds(run: str) -> dict[str, dict]:
    out = {}
    f = RES / run / "preds.ndjson"
    with (open(f) if f.exists() else gzip.open(f.with_suffix(".ndjson.gz"), "rt")) as f:
        for line in f:
            ev = json.loads(line)
            if ev.get("type") == "result":
                boxes = ev["products"]["identify"]["boxes"]
                best = max(boxes, key=lambda b: b["score"]) if boxes else None
                top = (best.get("species") or {}).get("top") or [] if best else []
                out[ev["path"]] = {"boxes": boxes, "best": best, "top": top, "names": [norm(t["scientific"]) for t in top]}
    return out


def pct(a: int, n: int) -> str:
    return f"{100 * a / n:.1f} %" if n else "-"


def species() -> None:
    gt = truth()
    for kind in ("bird", "mammal"):
        rows = [p for p, r in gt.items() if r["kind"] == kind]
        for ref, others in (("teacher", ["slim-fp16", "slim-fp16-half", "int8", "int8-half"]),
                            ("teacher-no-geo", ["slim-fp16-no-geo", "slim-fp16-half-no-geo", "int8-no-geo", "int8-half-no-geo"])):
            if not (RES / ref).exists():
                continue
            a = preds(ref)
            print(f"\n## {kind}s (n = {len(rows)}): each run against `{ref}`, photo by photo\n")
            print("| run | same first name | same top-5 in order | same top-5 set | posterior of first name: max / mean abs diff | right -> wrong | wrong -> right |")
            print("|---|---|---|---|---|---|---|")
            for o in others:
                if not (RES / o).exists():
                    print(f"| {o} | untested | | | | | |")
                    continue
                b = preds(o)
                same1 = sum(a[p]["names"][:1] == b[p]["names"][:1] for p in rows)
                same5 = sum(a[p]["names"][:5] == b[p]["names"][:5] for p in rows)
                set5 = sum(set(a[p]["names"][:5]) == set(b[p]["names"][:5]) for p in rows)
                d = [abs(a[p]["top"][0]["posterior"] - b[p]["top"][0]["posterior"]) for p in rows
                     if a[p]["top"] and b[p]["top"] and a[p]["names"][0] == b[p]["names"][0]]
                t = [norm(gt[p]["scientific"]) for p in rows]
                ra = [a[p]["names"][:1] == [s] for p, s in zip(rows, t)]
                rb = [b[p]["names"][:1] == [s] for p, s in zip(rows, t)]
                print(f"| {o} | {same1} | {same5} | {set5} | {max(d):.4f} / {statistics.mean(d):.5f} | "
                      f"{sum(x and not y for x, y in zip(ra, rb))} | {sum(y and not x for x, y in zip(ra, rb))} |")
    print("\n(posterior differences over photos where both runs put the same name first)")
    print("\n## Rank above species: first name's genus / family is the truth's\n")
    print("| run | kind | n | species | genus | family (truth's family known) |")
    print("|---|---|---|---|---|---|")
    fam = family_of()
    for run in ("teacher", "int8", "int8-half", "teacher-no-geo", "int8-no-geo", "int8-half-no-geo"):
        if not (RES / run).exists():
            continue
        a = preds(run)
        for kind in ("bird", "mammal"):
            rows = [p for p, r in gt.items() if r["kind"] == kind]
            t = {p: gt[p]["scientific"] for p in rows}
            sp = sum(bool(a[p]["top"]) and a[p]["names"][0] == norm(t[p]) for p in rows)
            ge = sum(bool(a[p]["top"]) and a[p]["names"][0].split(" ")[0] == norm(t[p]).split(" ")[0] for p in rows)
            known = [p for p in rows if norm(t[p]) in fam]
            fa = sum(bool(a[p]["top"]) and a[p]["top"][0]["taxonomy"][4].lower() == fam[norm(t[p])] for p in known)
            print(f"| {run} | {kind} | {len(rows)} | {pct(sp, len(rows))} | {pct(ge, len(rows))} | {pct(fa, len(known))} (n = {len(known)}) |")


def family_of() -> dict[str, str]:
    """Scientific name -> family, from the taxonomy every predicted name carries (all runs' top fives)."""
    out = {}
    for run in ("teacher", "teacher-no-geo"):
        if (RES / run).exists():
            for v in preds(run).values():
                for t in v["top"]:
                    out[norm(t["scientific"])] = t["taxonomy"][4].lower()
    return out


def iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def boxes() -> None:
    gt = truth()
    rows = [p for p, r in gt.items() if r["kind"] == "bird"]
    ref = preds("teacher")
    print(f"## Bird photos (n = {len(rows)}): each detector's best box against OWLv2's (run `teacher`); no true boxes exist\n")
    print("| detector | run | photos with a box | boxes per photo (mean) | best-box IoU with OWLv2: mean / >= 0.5 / >= 0.9 (both have a box) | best box's share of the frame: p10 / median / p90 |")
    print("|---|---|---|---|---|---|")
    for name, run in (("OWLv2", "teacher"), ("OWLv2 int8", "det-owlv2-int8"), ("RT-DETRv2 r50", "det-rtdetr-r50"), ("RT-DETRv2 r18", "det-rtdetr-r18"),
                      ("D-FINE small", "det-dfine-small"), ("D-FINE nano", "det-dfine-nano"), ("YOLOv10 m", "det-yolov10m"),
                      ("YOLOv10 s", "det-yolov10s"), ("YOLOv10 n", "det-yolov10n")):
        a = preds(run)
        has = [p for p in rows if a[p]["best"]]
        both = [p for p in has if ref[p]["best"]]
        ious = [iou(a[p]["best"]["xyxy"], ref[p]["best"]["xyxy"]) for p in both]
        area = sorted((lambda x: (x[2] - x[0]) * (x[3] - x[1]))(a[p]["best"]["xyxy"]) for p in has)
        q = lambda f: area[min(len(area) - 1, int(f * len(area)))]   # noqa: E731
        print(f"| {name} | `{run}` | {len(has)} | {statistics.mean(len(a[p]['boxes']) for p in rows):.2f} | "
              f"{statistics.mean(ious):.3f} / {pct(sum(i >= 0.5 for i in ious), len(ious))} / {pct(sum(i >= 0.9 for i in ious), len(ious))} (n = {len(ious)}) | "
              f"{q(0.1):.3f} / {q(0.5):.3f} / {q(0.9):.3f} |")
    print("\nBoxes per photo counts every box the engine kept (any kind); the COCO detectors report birds only.")


if __name__ == "__main__":
    {"species": species, "boxes": boxes}[sys.argv[1]]()
