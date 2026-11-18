#!/bin/zsh
# Bird crops + teacher vectors for a tree of photos (SPECIES/file.jpg), cut exactly as gardener cuts them at
# identify time (its detector, its crop framing), so a student trains on what it will be fed.
#   ./crops.sh data/inat-raw data/inat-crops [PER_SPECIES]     # -> OUT/img/SPECIES/*.jpg, files.txt, vecs.npy (embed.py's layout)
# Golden photos are skipped (embed.py's leaked()). Resumable: a species with OUT/rec/SPECIES is not redone.
# Photos with no animal found give no crop: that is the filter for feathers, eggs, nests and empty frames.
cd "$(dirname "$0")"; SRC=$(cd $1 && pwd); OUT=$PWD/$2; N=${3:-0}; B=$(cd ../bioscan && pwd); mkdir -p $OUT/rec $OUT/tmp $OUT/src
curl -s -m 2 http://127.0.0.1:8772/health >/dev/null && { echo "port 8772 is busy"; exit 1; }
./run - $SRC $OUT $N <<'PY' || exit 1
import os, sys
from pathlib import Path
from distill.embed import EXTS, golden_ids, leaked
src, out, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
gold, ids = golden_ids()
for d in sorted(p for p in src.iterdir() if p.is_dir()):
    drop = leaked(src, d.name, ids) if d.name in gold else set()
    keep = [f for f in sorted(d.iterdir()) if f.suffix.lower() in EXTS and f.stem not in drop]
    (out / "src" / d.name).mkdir(exist_ok=True)
    for f in keep[:n] if n else keep:
        if not (out / "src" / d.name / f.name).exists():
            os.link(f, out / "src" / d.name / f.name)
PY
PYTHONUNBUFFERED=1 ./run distill/serve.py --port 8772 --record $OUT/tmp --allow-root $OUT > $OUT/serve.log 2>&1 &
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8772/health >/dev/null && break; sleep 2; done
T0=$(date +%s); K=0
for D in $OUT/src/*(/); do
  S=$D:t; [[ -d $OUT/rec/$S ]] && continue
  rm -f $OUT/tmp/*(N)
  (cd $B && uv run -q gardener-engine --url http://127.0.0.1:8772 run $D --no-geo --json --out /dev/null) >> $OUT/run.log 2>&1
  mkdir $OUT/rec.part && mv $OUT/tmp/*(N) $OUT/rec.part/ 2>/dev/null; mv $OUT/rec.part $OUT/rec/$S
  K=$((K+1)); (( K % 50 == 0 )) && echo "$K species, $(( $(date +%s) - T0 )) s"
done
pkill -f "distill/serve.py --port 8772"
./run - $OUT <<'PY'
import sys
from pathlib import Path
import numpy as np
out = Path(sys.argv[1])
files, vecs, photos = [], [], 0
for d in sorted((out / "rec").iterdir()):
    photos += len(list((out / "src" / d.name).iterdir()))
    for npy in sorted(d.glob("[0-9]*.npy")):
        v = np.load(npy)
        for i in range(len(v)):
            files.append(f"{d.name}/{npy.stem}_{i:02d}.jpg")
        vecs.append(v)
    (out / "img" / d.name).mkdir(parents=True, exist_ok=True)
    for j in d.glob("*.jpg"):
        if not (out / "img" / d.name / j.name).exists():
            j.link_to(out / "img" / d.name / j.name) if hasattr(j, "link_to") else (out / "img" / d.name / j.name).hardlink_to(j)
(out / "files.txt").write_text("\n".join(files) + "\n")
np.save(out / "vecs.npy", np.concatenate(vecs))
print(f"done: {len(files)} crops from {photos} photos, {len(list((out / 'rec').iterdir()))} species")
PY
