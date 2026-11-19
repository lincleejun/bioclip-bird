"""Download the evaluation photos listed in eval/groundtruth.csv from iNaturalist (stdlib only).

  python eval/fetch_photos.py                  # eval/groundtruth.csv -> eval/photos/SPECIES/OBSERVATION_PHOTO.jpg
  python eval/fetch_photos.py inat2026.csv     # the second set -> eval/photos2026/...
  python eval/fetch_photos.py inat-global.csv  # the worldwide set -> eval/photosglobal/...; rerun any to resume

The photos are not redistributed here: each keeps its own licence (license and attribution columns). File names
are OBSERVATION_PHOTO.jpg; the script asks the iNaturalist API for those observations (200 per call), takes the
photo with that id at "medium" size (500 px, what the evaluation used) and checks it against the sha256 column.
One request a second, API and photos alike. A photo that is gone or whose file has changed is reported, not kept.
Checked against the live API on 3 photos (2026-11-17); the full list has not been re-downloaded.
"""
from __future__ import annotations

import csv
import hashlib
import sys
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "https://api.inaturalist.org/v1/observations"
UA = "bioclip-bird-eval-fetch/0.1"
last = [0.0]


def get(url: str) -> bytes:
    time.sleep(max(0.0, 1.0 - (time.monotonic() - last[0])))
    last[0] = time.monotonic()
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
        return r.read()


def main() -> None:
    with open(HERE / (sys.argv[1] if len(sys.argv) > 1 else "groundtruth.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f) if not (HERE / r["path"]).exists()]
    ids = {r["path"]: Path(r["path"]).stem.split("_") for r in rows}
    obs = sorted({o for o, _ in ids.values()})
    photos = {}
    for i in range(0, len(obs), 200):
        for o in json.loads(get(f"{API}?id={','.join(obs[i:i + 200])}&per_page=200"))["results"]:
            for p in o.get("photos") or []:
                photos[str(p["id"])] = p
    bad = []
    for n, r in enumerate(rows, 1):
        p = photos.get(ids[r["path"]][1])
        if not p or not p.get("url"):
            bad.append((r["path"], "photo not found"))
            continue
        data = get(p["url"].replace("/square.", "/medium."))
        if hashlib.sha256(data).hexdigest() != r["sha256"]:
            bad.append((r["path"], "file differs from the one evaluated"))
            continue
        out = HERE / r["path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        if n % 100 == 0:
            print(f"{n}/{len(rows)}", flush=True)
    for path, why in bad:
        print(f"missing: {path}: {why}")
    print(f"done: {len(rows) - len(bad)} downloaded, {len(bad)} missing")


if __name__ == "__main__":
    main()
