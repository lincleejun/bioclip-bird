"""Quality report for an embed.py folder whose species folders are trusted labels (iNaturalist research grade).

  ./run distill/quality.py data/inat            # report only
  ./run distill/quality.py data/inat --write    # also files.txt/vecs.npy without the dropped rows (originals kept as *.all)

Uses the teacher's own vectors, no model run. A photo is dropped when the teacher sees nothing of its label in it
(label outside the teacher's top 50 of 11,172 birds: feathers, eggs, nests, tracks, habitat shots, wrong uploads), or
when it is a copy of an earlier photo of the same species (cosine > 0.995). Nothing under img/ or the source is touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from distill.data import BIRDS, NAMES

out = Path(sys.argv[1])
files = (out / ("files.txt.all" if (out / "files.txt.all").exists() else "files.txt")).read_text().splitlines()
vecs = np.load(out / ("vecs.all.npy" if (out / "vecs.all.npy").exists() else "vecs.npy"))
z = np.load(NAMES / BIRDS, allow_pickle=False)
row = {s.replace(" ", "_"): i for i, s in enumerate(z["scientific"])}
label = np.array([row.get(f.split("/")[0], -1) for f in files])
rank = np.full(len(files), -1)
for i in range(0, len(files), 4096):
    s = vecs[i:i + 4096].astype(np.float32) @ z["matrix"].T
    k = label[i:i + 4096]
    rank[i:i + 4096] = np.where(k >= 0, (s > s[np.arange(len(s)), k][:, None]).sum(1), -1)
dup = np.zeros(len(files), bool)
start = 0
for end in [*np.flatnonzero(label[1:] != label[:-1]) + 1, len(files)]:       # files.txt is sorted by species
    v = vecs[start:end].astype(np.float32)
    dup[start:end] = (np.triu(v @ v.T, 1) > 0.995).any(0)
    start = end
known = label >= 0
print(f"{len(files)} photos, {len(set(f.split('/')[0] for f in files))} species")
print(f"label not in the name list: {(~known).sum()} photos, {len({f.split('/')[0] for f, k in zip(files, known) if not k})} species (kept: the vector is still the teacher's)")
for n in (1, 5, 50):
    print(f"label in teacher's top {n} (no location): {(rank[known] < n).mean():.1%}")
drop = (known & (rank >= 50)) | dup
print(f"dropped: {(known & (rank >= 50)).sum()} label outside top 50, {dup.sum()} copies, {drop.sum()} in all ({drop.mean():.1%})")
(out / "dropped.txt").write_text("".join(f"{f}\t{'copy' if d else 'rank ' + str(r)}\n" for f, r, d, x in zip(files, rank, dup, drop) if x))
if "--write" in sys.argv:
    if not (out / "files.txt.all").exists():
        (out / "files.txt").rename(out / "files.txt.all")
        (out / "vecs.npy").rename(out / "vecs.all.npy")
    (out / "files.txt").write_text("\n".join(f for f, x in zip(files, drop) if not x) + "\n")
    np.save(out / "vecs.npy", vecs[~drop])
    print(f"wrote {int((~drop).sum())} rows")
