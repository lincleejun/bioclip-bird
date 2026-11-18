# Results

Two separate questions, measured separately:

- **A. Naming the bird**: the species model (BioCLIP 2.5 and its smaller files) given a bird, which species is it.
- **B. Finding the bird**: the detector (OWLv2 and the alternatives), where in the photo is the bird.

Each has its own size and speed section (A6, B5). Only one part changes per comparison; what stays fixed is said
above each table. Every number names the directory under `eval/results/` it comes from; "untested" means the run
was not made. The evaluation set (1,571 iNaturalist photos: 1,046 birds of 42 species, 525 mammals of 22, mostly
California) is described in [data.md](data.md).

**Pipeline runs** (A1-A3, B2, B4) are gardener's whole engine on each photo: a SigLIP2 whole-frame gate, the
detector, BioCLIP 2.5 on each box's square crop against 11,172 AviList bird names (or MDD mammal names), and the
BirdNET location prior from the photo's coordinates unless the run is `-no-geo`. *Top-1 / Top-5* = the true
species is first / in the first five for the highest-scoring box; a photo without a box counts as a miss.
*Coverage* = the best box is named at species level (the engine's confidence rule); *precision* = top-1 among
those. Columns are copied from each run's `report.md`.

## A. Naming the bird

### A1. Original, fp16 and int8, through the whole pipeline

Fixed: gate, OWLv2 detector, names matrix. Changed: the BioCLIP 2.5 weights file and the compute precision.

| Species model | Location prior | Birds top-1 / top-5 | Birds coverage / precision | Mammals top-1 / top-5 | Directory |
|---|---|---|---|---|---|
| original (fp32, 3.94 GB) | yes | 92.4 % / 96.9 % | 93.6 % / 97.0 % | 91.4 % / 95.4 % | `teacher` |
| fp16 file, fp32 compute | yes | 92.4 % / 96.9 % | 93.6 % / 97.0 % | 91.4 % / 95.4 % | `slim-fp16` |
| fp16 file, fp16 compute | yes | 92.4 % / 96.9 % | 93.6 % / 97.0 % | 91.4 % / 95.4 % | `slim-fp16-half` |
| int8 file, fp32 compute | yes | 92.4 % / 96.9 % | 93.7 % / 97.0 % | 91.6 % / 95.4 % | `int8` |
| int8 file, fp16 compute | yes | 92.4 % / 96.9 % | 93.7 % / 97.0 % | 91.6 % / 95.4 % | `int8-half` |
| original (fp32) | no | 87.0 % / 96.1 % | 93.3 % / 91.9 % | 85.7 % / 94.3 % | `teacher-no-geo` |
| fp16 file, fp32 compute | no | 87.0 % / 96.1 % | 93.3 % / 91.9 % | 85.7 % / 94.3 % | `slim-fp16-no-geo` |
| fp16 file, fp16 compute | no | 87.0 % / 96.1 % | 93.4 % / 91.8 % | 85.7 % / 94.3 % | `slim-fp16-half-no-geo` |
| int8 file, fp32 compute | no | 87.0 % / 96.1 % | 93.5 % / 91.8 % | 85.9 % / 94.3 % | `int8-no-geo` |
| int8 file, fp16 compute | no | 87.2 % / 96.1 % | 93.5 % / 91.8 % | 85.9 % / 94.3 % | `int8-half-no-geo` |

The location prior is worth 5.4 points of bird top-1 here (92.4 % against 87.0 %); top-5 hardly moves.
All ten runs have the same best box on every bird photo (checked with `eval/agree.py`'s loader), so the rows differ
only by the species model.

### A2. Photo by photo against the original

Same runs as A1, compared photo by photo with `python eval/agree.py species` (`agree-species.md`). Posterior
differences are over photos where both put the same name first.

| Birds (n = 1,046) | Same first name | Same top-5, in order | Same top-5, as a set | Posterior of the first name: max / mean difference | Right -> wrong | Wrong -> right |
|---|---|---|---|---|---|---|
| fp16, fp32 compute | 1,046 | 1,046 | 1,046 | 0 / 0 | 0 | 0 |
| fp16, fp16 compute | 1,046 | 1,017 | 1,038 | 0.063 / 0.0004 | 0 | 0 |
| int8, fp32 compute | 1,045 | 890 | 976 | 0.118 / 0.0018 | 0 | 1 |
| int8, fp16 compute | 1,045 | 882 | 974 | 0.119 / 0.0018 | 0 | 1 |
| *no location prior, against `teacher-no-geo`:* | | | | | | |
| fp16, fp32 compute | 1,046 | 1,046 | 1,046 | 0 / 0 | 0 | 0 |
| fp16, fp16 compute | 1,046 | 1,007 | 1,024 | 0.060 / 0.0007 | 0 | 0 |
| int8, fp32 compute | 1,044 | 891 | 987 | 0.108 / 0.0031 | 1 | 1 |
| int8, fp16 compute | 1,040 | 887 | 985 | 0.123 / 0.0030 | 2 | 4 |

Mammals (n = 525): int8 keeps the same first name on 523 (with prior) and 524 / 523 (without); every change is
in `agree-species.md`. Where the order of the top five changes, it is mostly among names far below the first.

### A3. Above species, and the usual mistakes

First name's genus and family against the truth's (`agree-species.md`):

| Run | Birds species / genus / family | Mammals species / genus / family |
|---|---|---|
| original (`teacher`) | 92.4 % / 94.9 % / 96.0 % | 91.4 % / 94.3 % / 94.7 % |
| int8 (`int8`) | 92.4 % / 95.0 % / 96.0 % | 91.6 % / 94.1 % / 94.5 % |
| original, no prior (`teacher-no-geo`) | 87.0 % / 94.3 % / 95.8 % | 85.7 % / 92.4 % / 93.7 % |
| int8, no prior (`int8-no-geo`) | 87.0 % / 94.4 % / 95.8 % | 85.9 % / 92.4 % / 93.7 % |

The most frequent bird confusions are nearly the same for the original and int8 (each `report.md`): *Tyto
furcata* named *T. alba* 18 times in both, no box on 4 burrowing owls, *Nycticorax nycticorax* named *Nyctanassa
violacea* 3 times, then pairs of 2; the original's list also has *Astur cooperii* named *Accipiter striatus* twice,
which int8's top ten does not.

### A4. The species model alone, on crops

No detector, gate or location prior: one fixed crop per bird photo (the original pipeline's best box, gardener's
framing), image against the 11,172 names. `./run eval/crops.py` -> `crops/report.md`.

| Model (1,039 bird crops) | Top-1 | Top-5 | Cosine to the original's vector: mean / min | Same first name as the original |
|---|---|---|---|---|
| BioCLIP 2.5 original (fp32) | 87.8 % | 96.9 % | 1 / 1 | - |
| fp16 file, fp32 compute | 87.8 % | 96.9 % | 1.00000 / 1.00000 | 1,039 |
| fp16 file, fp16 compute | 87.8 % | 96.9 % | 0.99999 / 0.99977 | 1,039 |
| int8 file, fp32 compute | 87.8 % | 96.9 % | 0.99983 / 0.98226 | 1,037 |
| int8 file, fp16 compute | 87.9 % | 96.9 % | 0.99982 / 0.98222 | 1,034 |
| BioCLIP 1 (`imageomics/bioclip`, ViT-B/16, 0.6 GB) | 51.4 % | 74.8 % | other vector space | - |

BioCLIP 1 ran offline from the local cache with its own text tower encoding the same 11,172 names as
`a photo of <7 ranks> with common name <name>.`; that one template is our choice (BioCLIP 2.5's names use
TreeOfLife's precomputed vectors), so read its row as indicative.

The 3,171 crops whose vectors the original recorded during its pipeline run (`data/rec/golden`, birds and
mammals) were saved as 448 px JPEGs, so no model can reproduce the recorded vectors exactly. Against them
(`crops/report.md`): original on the saved JPEGs, cosine mean 0.99118, same first bird name 2,882 of 3,171;
fp16 0.99118 / 2,882 (fp32 compute), 0.99118 / 2,881 (fp16 compute); int8 0.99105 / 2,871 (both).

**Names matrix stored as float16** (the released file): on those 3,171 recorded vectors the first name is
unchanged for all 3,171 and the top-5 set for 3,166; similarities change by at most 4e-5 (see MODELS.md).

**Core ML** (`mobile/`, another quantisation: coremltools' linear 8-bit weights per output channel, fp16 compute,
633.9 MB; numbers from `mobile/README.md`, run on this Mac, not a phone): on 600 recorded bird crops, against the
same weights in PyTorch on the same input, cosine mean 0.99963 (min 0.99586) and the same first name on 592 of
600; against the recorded original vectors 562 of 600, where PyTorch itself gets 564.

### A5. Other models in BioCLIP's seat: distilled students

A student is a small image encoder trained to reproduce the original's image vector on whole photos (loss
1 - cosine), dropped into the pipeline with the same names matrix. Fixed: gate, OWLv2, names, prior. Every
species whose name hash falls in one of 20 buckets (492 species) is never trained on.

| Species model | Params | File | Training photos | Birds top-1 / top-5 | No prior: top-1 / top-5 | Directory |
|---|---|---|---|---|---|---|
| BioCLIP 2.5 original | 632 M (image tower) | 3.94 GB | - | 92.4 % / 96.9 % | 87.0 % / 96.1 % | `teacher`, `teacher-no-geo` |
| ViT-S/16 (`s-small-1`) | 22 M | 88 MB | 116k local photos | 37.7 % / 49.4 % | 16.9 % / 35.1 % | `s-small-1[-no-geo]` |
| ViT-B/16 from SigLIP2 (`s-medium-1`) | 94 M | 375 MB | 116k local photos | 42.1 % / 55.9 % | 21.7 % / 40.7 % | `s-medium-1[-no-geo]` |
| ViT-S/16 (`s-small-2`) | 22 M | 88 MB | + 183k iNaturalist photos | 60.0 % / 78.7 % | 45.2 % / 69.1 % | `s-small-2[-no-geo]` |
| ViT-B/16 + iNaturalist photos | 94 M | | | untested | untested | |

Agreement with the original on its recorded crops (last epoch of `train.py`): same first species 7 % / 9 % / 22 %
(cosine 0.52 / 0.56 / 0.63) for s-small-1 / s-medium-1 / s-small-2; on photos of the held-out species 46 % / 48 % /
45 % (cosine 0.85 / 0.86 / 0.86). **The students are far from the original** (the best is 32 points behind on
top-1). More photos helped most (37.7 % -> 60.0 %), but those photos come from iNaturalist like the evaluation
set, so part of the gain may be closeness to the test photos. Student weights are not released. Training:
`distill/train.py` (ViT-S: 10 epochs, batch 128, lr 2e-4, from `timm/vit_small_patch16_224.augreg_in21k_ft_in1k`;
ViT-B: 5 epochs, batch 64, lr 5e-5, from SigLIP2 base).

### A6. Size and speed of the species model

| File | Size | Format | Params |
|---|---|---|---|
| original checkpoint (`imageomics/bioclip-2.5-vith14`, both towers) | 3.94 GB | open_clip, fp32 | 632 M image tower + text tower |
| `bioclip-2.5-vith14-visual-fp16.safetensors` | 1.26 GB | safetensors, fp16 | 632 M |
| `bioclip-2.5-vith14-visual-int8.safetensors` | 0.63 GB | safetensors, int8 weights + fp16 scales | 632 M |
| Core ML `BioCLIP.mlpackage` (mobile/) | 633.9 MB | Core ML, 8-bit linear | 632 M |

Speed of the image tower alone (GPU idle, batches of 32, Apple M4 Max; one measurement from the working notes, its
log not kept): 33 images/s computing in fp32, 40 images/s in fp16, the same for the fp16 and the int8 file (the
file format changes the size on disk, not the compute). Loading the tower from the slim file takes about 5 s
without reading the original checkpoint. Memory was not measured reliably: the peak resident size read 4.5 GB for
all four variants. For comparison, `distill/bench.py` measured the original's forward at 31.9 images/s
(*notes*). Core ML on this Mac: see A4 (speed not measured). **A dedicated speed test on an
idle machine (batch 1, CPU, Core ML) was not done.**

## B. Finding the bird

### B1. Against true boxes: CUB-200-2011 test split

Fixed: nothing else runs; each detector alone on 5,794 photos with one bird and one true box each.
`./run eval/cub_boxes.py NAME` -> `boxes-cub-NAME/` (per photo, gzipped). Thresholds as in the pipeline: OWLv2 keeps
"a bird" boxes at 0.3, the COCO detectors their bird boxes at 0.25; a photo with none takes boxes down to 0.1
(OWLv2) or 0.125 (COCO), as the engine's second ask does. The answer is the highest-scoring box. *Found* = any
box; IoU mean / median over photos with a box; IoU >= 0.5 / 0.75 over all photos (no box = 0).

| Detector | Found | IoU mean / median | IoU >= 0.5 | IoU >= 0.75 | Boxes per photo | Directory |
|---|---|---|---|---|---|---|
| D-FINE nano | 99.98 % | 0.892 / 0.934 | 98.3 % | 90.0 % | 1.53 | `boxes-cub-dfine-nano` |
| D-FINE small | 99.95 % | 0.903 / 0.942 | 98.7 % | 91.8 % | 1.30 | `boxes-cub-dfine-small` |
| OWLv2, OWLv2 int8, RT-DETRv2 r18 / r50, YOLOv10 n / s / m | untested | | | | | |

The other seven were not run: the run was stopped before it reached them.

Limits: CUB photos are close-ups of one bird, mostly well framed: easier than real shoots, and they say nothing
about small or distant birds or flocks. The COCO detectors were trained on COCO's
"bird" class, not on CUB; OWLv2 is an open-vocabulary detector asked "a bird". CUB's boxes are drawn tight around the bird, which
favours detectors whose boxes are tight.

### B2. On the evaluation set: agreement with OWLv2 (no true boxes)

The 1,046 bird photos have no true boxes, so this measures how far each detector's best box is from OWLv2's,
not accuracy (`python eval/agree.py boxes` -> `agree-boxes.md`; the pipeline runs of B4).

| Detector | Photos with a box | Boxes per photo | Best-box IoU with OWLv2: mean / >= 0.5 / >= 0.9 | Best box's share of the frame: p10 / median / p90 | Directory |
|---|---|---|---|---|---|
| OWLv2 (reference) | 1,039 | 1.49 | 1 / 100 % / 100 % | 0.006 / 0.075 / 0.375 | `teacher` |
| OWLv2 int8 | 1,039 | 1.47 | 0.971 / 98.7 % / 98.3 % | 0.006 / 0.075 / 0.375 | `det-owlv2-int8` |
| RT-DETRv2 r50 | 1,032 | 1.87 | 0.833 / 88.9 % / 70.2 % | 0.007 / 0.076 / 0.374 | `det-rtdetr-r50` |
| RT-DETRv2 r18 | 1,026 | 2.20 | 0.806 / 86.9 % / 64.3 % | 0.006 / 0.075 / 0.374 | `det-rtdetr-r18` |
| D-FINE small | 1,026 | 2.28 | 0.809 / 86.9 % / 66.3 % | 0.007 / 0.075 / 0.371 | `det-dfine-small` |
| D-FINE nano | 1,027 | 2.35 | 0.787 / 85.4 % / 60.3 % | 0.007 / 0.075 / 0.387 | `det-dfine-nano` |
| YOLOv10 m | 956 | 1.23 | 0.842 / 89.7 % / 72.7 % | 0.008 / 0.080 / 0.373 | `det-yolov10m` |
| YOLOv10 s | 932 | 1.22 | 0.820 / 88.1 % / 68.2 % | 0.009 / 0.081 / 0.379 | `det-yolov10s` |
| YOLOv10 n | 864 | 1.08 | 0.802 / 87.2 % / 60.5 % | 0.010 / 0.090 / 0.384 | `det-yolov10n` |

Boxes per photo counts every box the engine kept; OWLv2's include the odd mammal or other-animal box. Half the
birds fill less than 7.5 % of the frame.

### B3. Small and distant birds (private shoot, summary only)

49 raw frames of a private shoot (2,048 px frames; not published, never trained on), with OWLv2's 94 bird boxes as
the reference: 12 very small (narrower than 5 % of the frame), 46 medium, 36 large. Boxes found by each detector at
the pipeline threshold (*notes*):

| Detector | Very small (12) | Medium (46) | Large (36) |
|---|---|---|---|
| D-FINE nano | 9 | 42 | 35 |
| D-FINE small | 7 | 44 | 34 |
| RT-DETRv2 r18 | 6 | 43 | 34 |
| RT-DETRv2 r50 | 8 | 44 | 34 |
| YOLOv10 n | 0 | 26 | 29 |
| YOLOv10 s | 0 | 36 | 32 |
| YOLOv10 m | 2 | 35 | 34 |

At threshold 0.1 D-FINE and RT-DETRv2 find 11-12 of the very small ones but add hundreds of other boxes (nano 790,
small 580, r18 336, r50 180), not checked for true or false.

### B4. Effect on naming the bird

Fixed: gate, original BioCLIP 2.5, location prior; only the detector changes. Birds, n = 1,046. The COCO
detectors keep only "bird", so they find almost no mammals (their mammal rows are in each report).

| Detector | Bird found | Top-1 | Top-5 | Directory |
|---|---|---|---|---|
| OWLv2 | 99.0 % | 92.4 % | 96.9 % | `teacher` |
| OWLv2 int8 | 99.0 % | 92.2 % | 96.9 % | `det-owlv2-int8` |
| RT-DETRv2 r50 | 98.5 % | 91.8 % | 96.2 % | `det-rtdetr-r50` |
| D-FINE small | 97.8 % | 91.2 % | 95.2 % | `det-dfine-small` |
| D-FINE nano | 97.9 % | 91.1 % | 95.0 % | `det-dfine-nano` |
| RT-DETRv2 r18 | 97.8 % | 90.7 % | 95.5 % | `det-rtdetr-r18` |
| YOLOv10 m | 91.2 % | 86.2 % | 89.6 % | `det-yolov10m` |
| YOLOv10 s | 89.0 % | 84.8 % | 87.6 % | `det-yolov10s` |
| YOLOv10 n | 82.5 % | 78.9 % | 81.5 % | `det-yolov10n` |

The OWLv2 int8 run loads its weights from the released file only; an earlier run that rounded the weights in
memory gave the same best box (to 3 decimals) and first species on all 1,571 photos.

### B5. Size and speed of the detectors

| Detector | Model | Licence | Size | Input | Runtime here |
|---|---|---|---|---|---|
| OWLv2 | `google/owlv2-base-patch16-ensemble` | Apache-2.0 | 620 MB | 960 x 960 (padded square) | transformers, MPS or CPU |
| OWLv2 int8 | released file | Apache-2.0 | 156 MB | 960 x 960 | transformers, MPS or CPU |
| RT-DETRv2 r50 / r18 | `PekingU/rtdetr_v2_r50vd` / `r18vd` | Apache-2.0 | 172 / 81 MB | 640 x 640 (processor) | transformers, MPS or CPU |
| D-FINE small / nano | `ustc-community/dfine-small-coco` / `nano-coco` | Apache-2.0 | 42 / 15 MB | 640 x 640 (processor) | transformers, MPS or CPU |
| YOLOv10 m / s / n | `onnx-community/yolov10m` / `s` / `n` | AGPL-3.0 | 62 / 29 / 9 MB | 640, letterboxed | onnxruntime, CPU only |
| D-FINE nano, Core ML (mobile/) | converted | Apache-2.0 | 15.5 MB | 640 x 640 | Core ML |

No dedicated detector speed test was done. What exists is the median "identify" time per bird photo that each
pipeline run of B4 logged in passing (gate + detector + original BioCLIP + prior, ms; each `report.md`): OWLv2 190,
OWLv2 int8 204, RT-DETRv2 r50 107, r18 107, D-FINE small 129, nano 124, YOLOv10 m 112, s 81, n 66. The GPU was
shared with other jobs to different degrees in each run, so these compare only roughly, and the YOLO models ran on
the CPU (onnxruntime) while the rest used the GPU (MPS). The CUB runs logged a median of 94 ms (D-FINE small) and
306 ms (D-FINE nano) per photo for the detector alone, both while other jobs were using the GPU heavily
(`boxes-cub-*/summary.json`): not a speed measurement. Core ML D-FINE nano: matches the transformers model on 80
evaluation photos (`mobile/README.md`), speed not measured.

## C. Whole pipeline, per photo (logged in passing)

The engine logs only two times per photo: decoding and "identify" (gate + detector + species + prior together).
They were taken while other jobs shared the machine to different degrees, so they compare poorly between runs;
the notes in A6 and B5 are all there is. Identify, median ms per photo, birds: original 190 (`teacher`), int8 184 / int8 fp16
compute 177, OWLv2 int8 204, D-FINE nano 124, RT-DETRv2 r50 107, YOLOv10 n 66 (CPU), students 141-149 (each
`report.md`). Decoding: 1-2 ms for these 500 px files.

## Other

**Beak/eye and flight heads** (weights and write-up in
[bioclip-bird-heads](https://github.com/lincleejun/bioclip-bird-heads), non-commercial only; both read block 15 of
the int8 tower). Beak and eyes on the CUB-200-2011 test split (5,794 images, scored once): median error 1.1 %
(beak) and 0.9 % (eye) of the crop's side, 97.8 % of beaks and 98.2 % of eyes within 5 %; its "is the part
visible" output is no better than always saying yes. Flying or not: 97.4 % right on 743 photos of species never
trained on, 97.5 % on 1,011 photos of photographers who posted both kinds.

**Aesthetic score.** gardener's is a ridge head on SigLIP2's whole-frame vector, fit on EVA (3,970 images). Same
data and 5-fold cross-validation with the mean of BioCLIP's block-15 tokens instead: Spearman 0.754 against 0.792
(`distill/aesthetic.py cv`; no result files published).

## Reproduce

The development scripts need [gardener](https://github.com/lincleejun/gardener) at `../bioscan` with its
evaluation photos; port 8771 free. `eval/agree.py` needs only this repository.

```
./eval_teacher.sh                                            # original               -> runs/teacher-golden
./run distill/slim.py models/bioclip-2.5-vith14-visual-int8.safetensors --int8
./run distill/serve.py --port 8771 --slim models/bioclip-2.5-vith14-visual-int8.safetensors [--half] \
    --allow-root $(cd ../bioscan && pwd)/data --allow-root $PWD/OUT &      # no --slim: the original
(cd ../bioscan && uv run gardener-engine --url http://127.0.0.1:8771 eval data/inat/groundtruth-inat.csv --out $OLDPWD/OUT [--no-geo])
./run distill/detectors.py                                   # writes models/owlv2-base-patch16-ensemble-int8.safetensors
./detbench.sh owlv2-int8 dfine-nano rtdetr-r50 yolov10n      # -> runs/det-NAME/golden
./eval_student.sh runs/s-small-2/student.pt [--no-geo]       # -> runs/s-small-2/golden[-no-geo]
python eval/agree.py species; python eval/agree.py boxes     # A2, A3, B2 from eval/results
./run eval/crops.py                                          # A4
./run eval/cub_boxes.py dfine-nano dfine-small ...          # B1, needs data/cub/CUB_200_2011
```

The `preds.ndjson[.gz]` files have photo paths rewritten to `photos/...` (relative to `eval/`); their header still
carries the sha256 of gardener's original CSV, which differs from `eval/groundtruth.csv`.
