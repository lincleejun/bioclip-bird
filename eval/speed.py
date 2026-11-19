"""Speed, load time and memory of each model on its own, one fresh process per measurement (Apple Silicon, MPS / CPU).

  ./run eval/speed.py all > eval/results/speed/raw.jsonl        # every measurement below, one after another
  ./mobile/ct eval/speed.py coreml >> eval/results/speed/raw.jsonl   # the Core ML conversions (needs coremltools)
  ./run eval/speed.py table < eval/results/speed/raw.jsonl > eval/results/speed/report.md

Species model: the image tower on random 224 x 224 input (preprocessing not included): batch 1 latency (median and
p90 of 30 calls) and batch 32 throughput (images/s over 10 s). Original = gardener's BioCLIP 2.5 (both towers loaded,
fp32 weights); fp16 / int8 = this repository's image-tower files, built from open_clip's config, nothing else loaded.
Detector: the first 100 bird photos of eval/groundtruth.csv, whole photo in, bird boxes out (the detector's own
preprocessing and post-processing included, image decoding not): per-photo median and p90 after 5 warm-up photos.
Memory: the process's peak resident size (ru_maxrss) and, on MPS, what the Metal driver had allocated at the end.
On Apple Silicon both share one pool, so neither alone is "the model's memory": read them together.
"""
from __future__ import annotations

import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ROOT.parent / "bioscan" / "data" / "inat"
SPECIES = [(v, d, p) for v in ("original", "fp16", "int8") for d, p in (("mps", "fp32"), ("mps", "fp16"), ("cpu", "fp32"))]
DETECTORS = [(n, d) for n in ("owlv2", "owlv2-int8", "rtdetr-r50", "rtdetr-r18", "dfine-small", "dfine-nano") for d in ("mps", "cpu")]
DETECTORS += [(n, "cpu") for n in ("yolov10m", "yolov10s", "yolov10n")]
COREML = [(m, u) for m in ("BioCLIP", "Detector") for u in ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL")]


def sync(dev: str) -> None:
    import torch
    if dev == "mps":
        torch.mps.synchronize()


def memory(dev: str) -> dict:
    out = {"peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)}
    if dev == "mps":
        import torch
        out["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
    return out


def pct(xs: list[float], q: float) -> float:
    return sorted(xs)[min(len(xs) - 1, int(q * len(xs)))]


def species(variant: str, dev: str, prec: str) -> dict:
    import torch
    t0 = time.perf_counter()
    if variant == "original":
        from gardener.service.adapters.bioclip import BioCLIP
        model = BioCLIP(dev).model.visual
    else:
        import open_clip
        from gardener.service.adapters.bioclip import snapshot_dir

        from distill import slim
        cfg = json.loads((Path(snapshot_dir()) / "open_clip_config.json").read_text())
        model = open_clip.CLIP(**cfg["model_cfg"]).visual
        model.load_state_dict(slim.load(str(ROOT / f"models/bioclip-2.5-vith14-visual-{variant}.safetensors")))
    dt = torch.float16 if prec == "fp16" else torch.float32
    model = model.eval().to(dev, dt)
    sync(dev)
    load = time.perf_counter() - t0
    x1, x32 = torch.randn(1, 3, 224, 224, device=dev, dtype=dt), torch.randn(32, 3, 224, 224, device=dev, dtype=dt)
    with torch.no_grad():
        for _ in range(3):
            model(x1)
        ms = []
        for _ in range(30):
            sync(dev)
            s = time.perf_counter()
            model(x1)
            sync(dev)
            ms.append(1000 * (time.perf_counter() - s))
        model(x32)
        sync(dev)
        n, s = 0, time.perf_counter()
        while time.perf_counter() - s < 10:
            model(x32)
            n += 32
        sync(dev)
        rate = n / (time.perf_counter() - s)
    return {"what": "species", "model": variant, "device": dev, "compute": prec, "load_s": round(load, 1),
            "b1_ms_median": round(statistics.median(ms), 1), "b1_ms_p90": round(pct(ms, 0.9), 1), "b32_img_s": round(rate, 1), **memory(dev)}


def photos(n: int = 100) -> list:
    import csv

    from PIL import Image
    rows = [r for r in csv.DictReader(open(ROOT / "eval" / "groundtruth.csv")) if r["kind"] == "bird"][:n]
    return [Image.open(PHOTOS / r["path"].removeprefix("photos/")).convert("RGB") for r in rows]


def detector(name: str, dev: str) -> dict:
    ims = photos()
    t0 = time.perf_counter()
    if name.startswith("owlv2"):
        from gardener.service.adapters.owlv2 import OWLv2

        from distill.detectors import owlv2_int8
        det = owlv2_int8(dev) if name == "owlv2-int8" else OWLv2(dev)
        run = lambda im: det.detect(im, ["a bird"], threshold=0.1)  # noqa: E731
    else:
        from distill.detectors import Coco
        det = Coco(name, dev, 0.25)
        run = det.boxes
    sync(dev)
    load = time.perf_counter() - t0
    for im in ims[:5]:
        run(im)
    ms = []
    for im in ims:
        sync(dev)
        s = time.perf_counter()
        run(im)
        sync(dev)
        ms.append(1000 * (time.perf_counter() - s))
    return {"what": "detector", "model": name, "device": dev, "load_s": round(load, 1), "photos": len(ims),
            "ms_median": round(statistics.median(ms), 1), "ms_p90": round(pct(ms, 0.9), 1), **memory(dev)}


def coreml(name: str, units: str) -> dict:
    import coremltools as ct
    import numpy as np
    from PIL import Image
    t0 = time.perf_counter()
    m = ct.models.MLModel(str(ROOT / "mobile" / "build" / f"{name}.mlpackage"), compute_units=getattr(ct.ComputeUnit, units))
    load = time.perf_counter() - t0
    side = 224 if name == "BioCLIP" else 640
    im = Image.fromarray(np.random.default_rng(0).integers(0, 255, (side, side, 3), dtype=np.uint8))
    out = m.predict({"image": im})
    zeros = all(float(np.abs(np.asarray(v)).max()) == 0 for v in out.values())
    for _ in range(3):
        m.predict({"image": im})
    ms = []
    for _ in range(30):
        s = time.perf_counter()
        m.predict({"image": im})
        ms.append(1000 * (time.perf_counter() - s))
    return {"what": "coreml", "model": name, "device": units, "load_s": round(load, 1), "b1_ms_median": round(statistics.median(ms), 1),
            "b1_ms_p90": round(pct(ms, 0.9), 1), "all_outputs_zero": zeros, **memory("cpu")}


def child(args: list[str]) -> None:
    """Run one measurement in a fresh process and pass its JSON line through; a failure becomes a line too."""
    r = subprocess.run([sys.executable, __file__, "one", *args], capture_output=True, text=True, cwd=ROOT)
    line = next((s for s in reversed(r.stdout.splitlines()) if s.startswith("{")), None)
    print(line or json.dumps({"args": args, "error": (r.stderr.strip().splitlines() or ["no output"])[-1][:300]}), flush=True)


def idle() -> str:
    """Load average at the start, so the report can say how quiet the machine was."""
    return json.dumps({"what": "machine", "loadavg": [round(x, 2) for x in __import__("os").getloadavg()]})


def table() -> None:
    rows = [json.loads(s) for s in sys.stdin if s.strip()]
    m = [r for r in rows if r.get("what") == "machine"]
    print("# Speed, load time and memory (Apple M4 Max, 64 GB)\n")
    print("Raw lines: `raw.jsonl`; method: `eval/speed.py`'s docstring. Machine at the start of each batch: "
          + "; ".join(f"load average {r['loadavg']}" for r in m) + ".\n")
    print("## Species model (image tower, random 224 px input, no preprocessing)\n")
    print("| model | device | compute | load s | batch 1 ms median / p90 | batch 32 images/s | peak RSS GB | MPS driver GB |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("what") == "species":
            print(f"| {r['model']} | {r['device']} | {r['compute']} | {r['load_s']} | {r['b1_ms_median']} / {r['b1_ms_p90']} | {r['b32_img_s']} | {r['peak_rss_gb']} | {r.get('mps_driver_gb', '-')} |")
    print("\n## Detector (100 bird photos, whole photo in, bird boxes out)\n")
    print("| detector | device | load s | ms per photo median / p90 | peak RSS GB | MPS driver GB |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        if r.get("what") == "detector":
            print(f"| {r['model']} | {r['device']} | {r['load_s']} | {r['ms_median']} / {r['ms_p90']} | {r['peak_rss_gb']} | {r.get('mps_driver_gb', '-')} |")
    cm = [r for r in rows if r.get("what") == "coreml"]
    if cm:
        print("\n## Core ML conversions on the Mac (not a phone; batch 1, model call only)\n")
        print("| model | compute units | load s (includes compiling) | ms median / p90 | outputs all zero | peak RSS GB |")
        print("|---|---|---|---|---|---|")
        for r in cm:
            print(f"| {r['model']} | {r['device']} | {r['load_s']} | {r['b1_ms_median']} / {r['b1_ms_p90']} | {'yes' if r['all_outputs_zero'] else 'no'} | {r['peak_rss_gb']} |")
    for r in rows:
        if "error" in r:
            print(f"\nFailed: {' '.join(r['args'])}: {r['error']}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "one":
        kind, *a = sys.argv[2:]
        print(json.dumps({"species": species, "detector": detector, "coreml": coreml}[kind](*a)), flush=True)
    elif cmd == "all":
        print(idle(), flush=True)
        for v, d, p in SPECIES:
            child(["species", v, d, p])
        for n, d in DETECTORS:
            child(["detector", n, d])
    elif cmd == "coreml":
        print(idle(), flush=True)
        for n, u in COREML:
            child(["coreml", n, u])
    elif cmd == "table":
        table()
