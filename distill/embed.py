"""Teacher vectors for a tree of photos (SPECIES/file.jpg), plus a small copy of each photo to train on.

  ./run distill/embed.py --src ../birding/spec/media --out data/birding

Writes OUT/img/SPECIES/name.jpg (short side 320 px), OUT/files.txt (one relative path per row) and
OUT/vecs.npy (float16, row i = the teacher's vector for files.txt line i). The teacher sees the
original through its own preprocessing. The source tree is only read. Photos of gardener's golden
set are skipped (by iNaturalist observation or photo id), so the eval set is never trained on.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

GOLDEN = Path(__file__).resolve().parents[2] / "bioscan" / "data" / "inat" / "groundtruth-inat.csv"
EXTS = {".jpg", ".jpeg", ".png"}
EDGE = 320


def golden_ids() -> tuple[set[str], set[str]]:
    """(species folder names, iNaturalist observation and photo ids) of the golden set."""
    with open(GOLDEN, newline="") as f:
        rows = list(csv.DictReader(f))
    ids = {part for r in rows for part in Path(r["path"]).stem.split("_")}
    return {r["scientific"].replace(" ", "_") for r in rows}, ids


@functools.lru_cache(maxsize=1)
def same_shoot(src: Path) -> frozenset[str]:
    """File stems in src/manifest.csv (fetch_inat.py) by a golden photo's photographer, of the same species
    on the same day: most likely the same bird under another observation id."""
    if not (src / "manifest.csv").exists():
        return frozenset()
    with open(GOLDEN, newline="") as f:
        gold = {(r["scientific"], r["attribution"].split(",")[0], r["taken_at"][:10]) for r in csv.DictReader(f)}
    with open(src / "manifest.csv", newline="") as f:
        return frozenset(Path(r["path"]).stem for r in csv.DictReader(f) if (r["scientific"], r["attribution"].split(",")[0], r["observed_on"]) in gold)


def leaked(src: Path, species: str, ids: set[str]) -> set[str]:
    """File stems under src/species that are golden photos: by the id in the name, or (GBIF rows) by
    the iNaturalist photo id in the manifest's asset URL; plus same_shoot()."""
    out = set(same_shoot(src))
    manifest = src.parent / "images" / f"{species}.jsonl"
    by_key = {}
    if manifest.exists():
        with open(manifest, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]     # not splitlines(): captions contain U+2028
        for r in rows:
            m = re.search(r"/photos/(\d+)/", r.get("asset") or "") or re.search(r"/photos/(\d+)", r.get("page") or "")
            if m:
                by_key[f"{r.get('source')}-{r.get('key')}"] = m.group(1)
    for f in (src / species).iterdir():
        num = f.stem.split("-", 1)[-1].split("-")[0]
        if num in ids or by_key.get(f.stem) in ids or ids & set(f.stem.split("_")):      # last: fetch_inat.py's OBSERVATION_PHOTO.jpg
            out.add(f.stem)
    return out


class Photos(torch.utils.data.Dataset):
    def __init__(self, src: Path, out: Path, rel: list[str], pre) -> None:
        self.src, self.out, self.rel, self.pre = src, out, rel, pre

    def __len__(self) -> int:
        return len(self.rel)

    def __getitem__(self, i: int):
        try:
            with Image.open(self.src / self.rel[i]) as im:
                im = im.convert("RGB")
                x = self.pre(im)
                small = self.out / "img" / Path(self.rel[i]).with_suffix(".jpg")
                if not small.exists():
                    small.parent.mkdir(parents=True, exist_ok=True)
                    k = EDGE / min(im.size)
                    if k > 0.6:                     # already small (iNaturalist's 500 px): link it, no second copy on disk
                        try:
                            os.link(self.src / self.rel[i], small)
                        except OSError:             # another volume
                            shutil.copyfile(self.src / self.rel[i], small)
                    else:
                        im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.BICUBIC).save(small, quality=90)
            return x, i
        except Exception:                           # a broken download: no vector, dropped from files.txt
            return torch.zeros(3, 224, 224), -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[], help="earlier --out folders of the same --src: their photos are not embedded again")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    src, out = Path(a.src).resolve(), Path(a.out)
    gold_species, ids = golden_ids()
    rel, skipped = [], 0
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        drop = leaked(src, d.name, ids) if d.name in gold_species else set()
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in EXTS and not f.name.startswith("."):
                if f.stem in drop:
                    skipped += 1
                else:
                    rel.append(f"{d.name}/{f.name}")
    have = {line for o in a.skip for line in (Path(o) / "files.txt").read_text().splitlines()}
    rel = [r for r in rel if str(Path(r).with_suffix(".jpg")) not in have]
    rel = rel[:a.limit] if a.limit else rel
    print(f"{len(rel)} photos, {skipped} golden photos skipped", flush=True)

    from gardener.service.adapters.bioclip import BioCLIP
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    teacher = BioCLIP(dev)
    loader = torch.utils.data.DataLoader(Photos(src, out, rel, teacher.preprocess), batch_size=32, num_workers=a.workers)
    vecs, keep, t0 = np.zeros((len(rel), 1024), np.float16), np.zeros(len(rel), bool), time.time()
    for n, (x, idx) in enumerate(loader):
        ok = idx >= 0
        if ok.any():
            with torch.no_grad():
                f = teacher.model.encode_image(x[ok].to(dev))
                f = f / f.norm(dim=-1, keepdim=True)
            vecs[idx[ok].numpy()] = f.float().cpu().numpy().astype(np.float16)
            keep[idx[ok].numpy()] = True
        if n % 200 == 0:
            done = (n + 1) * 32
            print(f"{min(done, len(rel))}/{len(rel)}  {done / (time.time() - t0):.1f} img/s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "files.txt").write_text("\n".join(str(Path(r).with_suffix(".jpg")) for r, k in zip(rel, keep) if k) + "\n")
    np.save(out / "vecs.npy", vecs[keep])
    (out / "meta.json").write_text(json.dumps({"src": str(src), "photos": int(keep.sum()), "failed": int((~keep).sum()), "golden_skipped": skipped,
                                               "teacher": "imageomics/bioclip-2.5-vith14", "licence": "mixed, personal research only (see the source tree's manifests)"}))
    print(f"done: {int(keep.sum())} vectors, {int((~keep).sum())} unreadable, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
