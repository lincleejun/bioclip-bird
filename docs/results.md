# Results

All runs score gardener's evaluation set: 1,571 research-grade iNaturalist photos (1,046 birds, 525 mammals),
listed in [`eval/groundtruth.csv`](../eval/groundtruth.csv). Each run's `report.md` and per-photo
`preds.ndjson.gz` are in `eval/results/<run>/`; every number below names its run. "Untested" means the run was
not made; nothing is filled in by guess.

**What is measured.** The whole gardener pipeline, with one part swapped at a time: a SigLIP2 whole-frame gate,
the detector (OWLv2 by default), BioCLIP 2.5 on each box's square crop against the 11,172-species AviList
matrix, and the BirdNET location prior from each photo's coordinates (`-no-geo` runs leave it out).
*Bird found* = a box on the photo (gardener's `detect` column); *Top-1 / Top-5* = the true species is first / in
the first five for the highest-scoring box; a photo without a box is a miss. Columns are copied from each
report's "Metrics by tier x kind" table.

**Timings** are the median "identify" time per photo that each evaluation logged in passing (Apple M4 Max,
MPS). Other jobs shared the GPU differently in each run, so read them only roughly; the YOLO detectors ran on the
CPU with onnxruntime.

## BioCLIP 2.5: original, fp16, int8

| Run | Species model | File | Birds top-1 / top-5 | Mammals top-1 / top-5 | Directory |
|---|---|---|---|---|---|
| original | `imageomics/bioclip-2.5-vith14`, fp32 | 3.94 GB (both towers) | 92.4 % / 96.9 % | 91.4 % / 95.4 % | `teacher` |
| fp16 file, fp32 compute | image tower only | 1.26 GB | 92.4 % / 96.9 % | 91.4 % / 95.4 % | `slim-fp16` |
| fp16 file, fp16 compute | image tower only | 1.26 GB | 92.4 % / 96.9 % | 91.4 % / 95.4 % | `slim-fp16-half` |
| int8 file, fp32 compute | image tower only | 0.63 GB | 92.4 % / 96.9 % | 91.6 % / 95.4 % | `int8` |
| int8 file, fp16 compute | image tower only | 0.63 GB | 92.4 % / 96.9 % | 91.6 % / 95.4 % | `int8-half` |
| int8 file, fp16 compute, **no location prior** | image tower only | 0.63 GB | 87.2 % / 96.1 % | 85.9 % / 94.3 % | `int8-half-no-geo` |

The original without the location prior: untested.

Speed of the image tower alone (GPU idle, batches of 32, one measurement, log not kept): 33 images/s computing
in fp32, 40 images/s in fp16, the same for the fp16 and the int8 file. Loading takes about 5 s. Memory: not
measured reliably (the peak read 4.5 GB for all four variants).

## Detectors in OWLv2's seat

Birds only (n = 1,046); everything else unchanged (original BioCLIP, location prior). The COCO detectors keep
only their "bird" class, so they find almost no mammals (their mammal rows are in each report). Threshold 0.25
mapped onto OWLv2's scale (`distill/detectors.py`); OWLv2 uses gardener's own 0.3.

| Detector | Size | Licence | Bird found | Top-1 | Top-5 | Identify ms (median) | Directory |
|---|---|---|---|---|---|---|---|
| OWLv2 base patch16 ensemble (gardener's) | 620 MB | Apache-2.0 | 99.0 % | 92.4 % | 96.9 % | 190 | `teacher` |
| OWLv2, int8 file | 156 MB | Apache-2.0 | 99.0 % | 92.2 % | 96.9 % | 204 | `det-owlv2-int8` |
| RT-DETRv2 r50 | 172 MB | Apache-2.0 | 98.5 % | 91.8 % | 96.2 % | 107 | `det-rtdetr-r50` |
| D-FINE small | 42 MB | Apache-2.0 | 97.8 % | 91.2 % | 95.2 % | 129 | `det-dfine-small` |
| D-FINE nano | 15 MB | Apache-2.0 | 97.9 % | 91.1 % | 95.0 % | 124 | `det-dfine-nano` |
| RT-DETRv2 r18 | 81 MB | Apache-2.0 | 97.8 % | 90.7 % | 95.5 % | 107 | `det-rtdetr-r18` |
| YOLOv10 m (ONNX, CPU) | 62 MB | AGPL-3.0 | 91.2 % | 86.2 % | 89.6 % | 112 | `det-yolov10m` |
| YOLOv10 s (ONNX, CPU) | 29 MB | AGPL-3.0 | 89.0 % | 84.8 % | 87.6 % | 81 | `det-yolov10s` |
| YOLOv10 n (ONNX, CPU) | 9 MB | AGPL-3.0 | 82.5 % | 78.9 % | 81.5 % | 66 | `det-yolov10n` |

Models: `google/owlv2-base-patch16-ensemble`, `PekingU/rtdetr_v2_r18vd` / `r50vd`,
`ustc-community/dfine-nano-coco` / `dfine-small-coco`, `onnx-community/yolov10n` / `s` / `m` (input 640).
The OWLv2 int8 run loads its weights from the released file only (`distill/detectors.py owlv2_int8()`; the
tokenizer and image settings still come from the upstream repository). An earlier run that rounded the weights
in memory gave the same best box (to 3 decimals) and the same first species on all 1,571 photos. Its two-tenths
of a point below the original OWLv2 on birds (and 90.9 % / 95.0 % on mammals against 91.4 % / 95.4 %) is the
int8 detector's own.

## Distilled students

A student is a small image encoder trained to reproduce the teacher's (original BioCLIP 2.5) image vector on
whole photos (loss 1 - cosine), then dropped into the teacher's seat with the same names matrix. Hold-out:
every species whose name hash falls in one of 20 buckets (492 species) is never trained on.

| Student | Params | File | Training photos | Birds top-1 / top-5 | No location prior | Directory |
|---|---|---|---|---|---|---|
| teacher (original BioCLIP 2.5) | 632 M (image tower) | 3.94 GB | - | 92.4 % / 96.9 % | untested | `teacher` |
| ViT-S/16 (`s-small-1`) | 22 M | 88 MB | 116k local photos | 37.7 % / 49.4 % | 16.9 % / 35.1 % | `s-small-1`, `s-small-1-no-geo` |
| ViT-B/16 from SigLIP2 (`s-medium-1`) | 94 M | 375 MB | 116k local photos | 42.1 % / 55.9 % | 21.7 % / 40.7 % | `s-medium-1`, `s-medium-1-no-geo` |
| ViT-S/16 (`s-small-2`) | 22 M | 88 MB | + 183k iNaturalist photos of US/CA/CN birds | 60.0 % / 78.7 % | 45.2 % / 69.1 % | `s-small-2`, `s-small-2-no-geo` |
| ViT-B/16 + iNaturalist photos | 94 M | | | untested | untested | |

Agreement with the teacher on the recorded evaluation crops (last epoch of training, `train.py`'s log): same
first species 7 % / 9 % / 22 % (cosine 0.52 / 0.56 / 0.63) for s-small-1 / s-medium-1 / s-small-2; on photos
of the held-out species 46 % / 48 % / 45 % (cosine 0.85 / 0.86 / 0.86).

**The students are far from the teacher**: the best is 32 points behind on top-1, and without the location
prior it is lower still. More photos helped most (37.7 % -> 60.0 % for the same small model), but the added
photos come from iNaturalist like the evaluation set, so part of that gain may be closeness to the test photos
rather than better birds. Student weights are not released.

Training: `distill/train.py` (ViT-S: 10 epochs, batch 128, lr 2e-4, from `timm/vit_small_patch16_224.augreg_in21k_ft_in1k`;
ViT-B: 5 epochs, batch 64, lr 5e-5, from SigLIP2 base).

## Beak/eye and flight heads

Weights and full write-up in [bioclip-bird-heads](https://github.com/lincleejun/bioclip-bird-heads)
(non-commercial only). Both read block 15's patch tokens of the int8 tower above.

- Beak and eyes, CUB-200-2011 test split (5,794 images, scored once): median error 1.1 % (beak) and 0.9 % (eye)
  of the crop's side; 97.8 % of beaks and 98.2 % of eyes within 5 %. Its "is the part visible" output is no
  better than always saying yes.
- Flying or not, iNaturalist observers' labels: 97.4 % right on 743 photos of species never trained on, 97.5 % on
  1,011 photos of photographers who posted both kinds.

## Aesthetic score (side note)

gardener's aesthetic score is a ridge head on SigLIP2's whole-frame vector, fit on EVA (3,970 images). Same data,
same 5-fold cross-validation, with the mean of BioCLIP's block-15 tokens instead: Spearman 0.754 against 0.792
(`distill/aesthetic.py cv`; no result files published). Usable, slightly worse.

## Reproduce

Needs [gardener](https://github.com/lincleejun/gardener) at `../bioscan` with its evaluation photos (its own copy
of this list, `data/inat/groundtruth-inat.csv`; `eval/groundtruth.csv` is the same 1,571 rows with paths
relative to `eval/` and a sha256 per photo, and `python eval/fetch_photos.py` downloads them). Port 8771 must be
free.

```
./eval_teacher.sh                                          # original           -> runs/teacher-golden
./run distill/slim.py models/bioclip-2.5-vith14-visual-int8.safetensors --int8
./run distill/serve.py --port 8771 --slim models/bioclip-2.5-vith14-visual-int8.safetensors [--half] --allow-root $(cd ../bioscan && pwd)/data --allow-root $PWD/OUT &
(cd ../bioscan && uv run gardener-engine --url http://127.0.0.1:8771 eval data/inat/groundtruth-inat.csv --out $OLDPWD/OUT [--no-geo])
./run distill/detectors.py                                 # writes models/owlv2-base-patch16-ensemble-int8.safetensors
./detbench.sh owlv2-int8 dfine-nano rtdetr-r50 yolov10n    # -> runs/det-NAME/golden
./eval_student.sh runs/s-small-2/student.pt [--no-geo]     # -> runs/s-small-2/golden[-no-geo]
```

The `preds.ndjson.gz` files here have photo paths rewritten to `photos/...` (relative to `eval/`); their header
still carries the sha256 of gardener's original CSV, which differs from `eval/groundtruth.csv`.
