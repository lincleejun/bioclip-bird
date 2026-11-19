# bioclip-bird

Smaller models for naming birds in photos, measured end to end on 1,571 iNaturalist photos.

> 中文简介：把 [gardener](https://github.com/lincleejun/gardener) 识鸟用的 BioCLIP 2.5（3.9 GB）和 OWLv2 检测器（620 MB）
> 压小：BioCLIP 只留看图的一半并存成 8 位（0.63 GB），OWLv2 存成 8 位（156 MB），评测集上成绩基本不变。
> 结果按两个维度分开写：认鸟（鸟种模型：原版 / 16 位 / 8 位、有无地点、逐张一致性、BioCLIP 第一代、蒸馏学生）和找鸟
> （检测模型：在 CUB 真值框上的框准度、与 OWLv2 的一致度、小鸟、对认鸟的影响），各自附体积与本机测速；另有数据集说明。
> 附 11,172 种鸟的名录矩阵、不依赖 gardener 的最小推理示例、全部评测清单和逐图结果。
> 评测有三套：加州 42 种（Gardener 原有）、美加中 228 种、全球 138 个国家 962 种（2026 年照片，随机抽种）。
> 全球集上原版第一名 77.2%，8 位版 77.4%，推荐组合 77.7% 且识别耗时约减半。
> 蒸馏出的小学生模型离原版差很远，只公开结果不发权重。
> 鸟眼鸟嘴和飞行两个小头在另一个仓库（权重仅限非商用）。

## What is here

1. **BioCLIP 2.5 image tower, int8** (0.63 GB, from 3.9 GB) and **fp16** (1.26 GB): the upstream weights
   without the text tower, rounded. Birds on the evaluation set: identical to the original (92.4 % / 96.9 %);
   mammals 91.6 % / 95.4 % against 91.4 % / 95.4 %.
2. **Bird names matrix**: 11,172 AviList v2026 species as BioCLIP 2.5 text vectors (22 MB), plus the logit
   scale (98.86). With the image tower it names a bird; no text tower needed.
3. **OWLv2 detector, int8** (156 MB, from 620 MB): same bird detection rate on the evaluation set; and the same
   without its text tower (92 MB), with the text vectors of gardener's 47 prompts stored in the file.
4. **Evaluation**: the 1,571-photo list with licences and a download script, every run's report and
   per-photo predictions, and the results write-up.

Weights are attached to the [GitHub Releases](https://github.com/lincleejun/bioclip-bird/releases), not in git.
File list, sizes, sha256, upstream models and licences: [MODELS.md](MODELS.md).

Add-on: **beak/eye points and flying-or-not** heads on the same tower live in
[bioclip-bird-heads](https://github.com/lincleejun/bioclip-bird-heads); their weights are for non-commercial use
only (trained on CUB-200-2011 and mostly CC BY-NC photos).

## Results

Full tables, sources and commands: [docs/results.md](docs/results.md). Data: [docs/data.md](docs/data.md).

### Naming the bird (species model)

Which species, given the bird. Everything else fixed: gardener's pipeline (SigLIP2 gate, OWLv2 detector, the
11,172-name matrix); only the BioCLIP 2.5 weights change. 1,046 bird photos.

| Species model | File | Top-1 / top-5, with location prior | Top-1 / top-5, image only (no prior) | Same first name as the original (with prior) |
|---|---|---|---|---|
| BioCLIP 2.5 original | 3.94 GB | 92.4 % / 96.9 % | 87.0 % / 96.1 % | - |
| fp16 image tower | 1.26 GB | 92.4 % / 96.9 % | 87.0 % / 96.1 % | 1,046 of 1,046 |
| int8 image tower | 0.63 GB | 92.4 % / 96.9 % | 87.0 % / 96.1 % | 1,045 of 1,046 |
| BioCLIP 1 (ViT-B/16), crops only | 0.6 GB | untested | 51.4 % / 74.8 % (crops, our name template) | - |
| Distilled ViT-S/16 student | 88 MB | 60.0 % / 78.7 % | 45.2 % / 69.1 % | - |

int8 changes no answer from right to wrong with the prior and one without; the students are far behind and not released.

**Worldwide** (`eval/inat-global.csv`: 2,632 photos of 962 species from 138 countries, all taken in 2026, species
drawn at random from every bird iNaturalist knows) the task is much harder: the original scores 77.2 % / 92.3 % with
the location prior and 66.4 % / 87.7 % without it; the int8 tower 77.4 % / 92.2 %. A second set of the US, Canada and
China (`eval/inat2026.csv`, 637 photos of 228 species) sits in between: 82.3 % / 92.8 % original, 82.6 % / 92.5 % int8.
Asia is the weakest region (image-only top-1 54 % worldwide; China 67 % on the second set).

**Location prior.** On the same crops of the worldwide set, GBIF record counts by country (state/province where the
table has them) and month, a small offline table, beat BirdNET geo: top-1 78.5 % against 75.7 %, top-5 93.8 % against
92.5 %, level or ahead in every region and on the least-observed species. On the US/Canada/China set the two tie.
Details and caveats: docs/results.md A7.

### Finding the bird (detector)

Where the bird is. Everything else fixed (gate, original BioCLIP, prior); only the detector changes. 1,046 bird
photos of the evaluation set, which have no true boxes: the IoU columns measure agreement with OWLv2's best box,
not accuracy. Against true boxes: the CUB-200-2011 test split (5,794 close-ups of one bird each).

| Detector | Size, licence | Bird found (eval set) | Best-box IoU with OWLv2: mean / >= 0.9 | CUB true boxes: IoU >= 0.5 / >= 0.75 |
|---|---|---|---|---|
| OWLv2 | 620 MB, Apache-2.0 | 99.0 % | reference | 99.2 % / 96.7 % |
| OWLv2 int8 | 156 MB, Apache-2.0 | 99.0 % | 0.971 / 98.3 % | 99.1 % / 96.5 % |
| RT-DETRv2 r50 | 172 MB, Apache-2.0 | 98.5 % | 0.833 / 70.2 % | 99.5 % / 94.6 % |
| RT-DETRv2 r18 | 81 MB, Apache-2.0 | 97.8 % | 0.806 / 64.3 % | 98.3 % / 90.1 % |
| D-FINE small | 42 MB, Apache-2.0 | 97.8 % | 0.809 / 66.3 % | 98.7 % / 91.8 % |
| D-FINE nano | 15 MB, Apache-2.0 | 97.9 % | 0.787 / 60.3 % | 98.3 % / 90.0 % |
| YOLOv10 m | 62 MB, AGPL-3.0 | 91.2 % | 0.842 / 72.7 % | 99.0 % / 93.7 % |
| YOLOv10 s | 29 MB, AGPL-3.0 | 89.0 % | 0.820 / 68.2 % | 98.1 % / 91.8 % |
| YOLOv10 n | 9 MB, AGPL-3.0 | 82.5 % | 0.802 / 60.5 % | 96.2 % / 86.2 % |

On CUB's close-ups every detector boxes the bird well; the YOLO models' lower finding rate on the evaluation set
comes from smaller, more distant birds.

The COCO detectors (all but OWLv2) find birds only, not mammals. Their effect on naming the bird is a separate
table in docs/results.md (B4): bird top-1 92.4 % with OWLv2, 92.2 % with OWLv2 int8, 91.1 % with D-FINE nano,
78.9 % with YOLOv10 n.

**Cascade: a small detector first, OWLv2 only when it finds nothing** (whole pipeline, same 1,046 bird photos):

| Detector | Bird found | Top-1 | Top-5 | Bird photos that needed OWLv2 |
|---|---|---|---|---|
| OWLv2 alone | 99.0 % | 92.4 % | 96.9 % | all |
| D-FINE nano, then OWLv2 int8 | 99.3 % | 92.3 % | 96.5 % | 19 of 1,046 |
| RT-DETRv2 r50, then OWLv2 int8 | 99.5 % | 92.5 % | 97.2 % | 14 of 1,046 |

Mammal photos go to OWLv2 as before (the COCO models answer only "a bird"). The fallback can be OWLv2 without its
text tower: 92 MB instead of 156 MB, same boxes (docs/results.md B4).

### Size and speed

Each model alone, one fresh process per measurement, Apple M4 Max 64 GB, PyTorch (MPS = the Mac's GPU).
Measured twice; the two passes agree within about 5 %, the numbers are the first pass. The machine was not idle
(another session was building in Xcode; load average 5-11). Full table, memory and method:
[eval/results/speed/report.md](eval/results/speed/report.md).

| Species model (image tower, 224 px) | File | MPS fp32: batch-1 ms / images/s at batch 32 | MPS fp16 | CPU fp32 | Peak memory (RSS) |
|---|---|---|---|---|---|
| BioCLIP 2.5 original | 3.94 GB | 47 ms / 32 | 41 ms / 39 | 197 ms / 7.6 | 4.4 GB |
| fp16 image tower | 1.26 GB | 46 ms / 33 | 42 ms / 39 | 199 ms / 7.6 | 5.2 GB |
| int8 image tower | 0.63 GB | 47 ms / 32 | 42 ms / 39 | 194 ms / 7.7 | 7.2 GB |
| int8, Core ML on the Mac (Neural Engine) | 0.63 GB | 71 ms (batch 1) | | 99 ms | 1.0 GB |

int8 is a storage format: it computes as fast as the original. The peak memory above is from before a loading
fix; loading the int8 file straight into fp16 now takes 1.6 s and peaks at 2.4 GB on the GPU path (same weights bit
for bit; A6). fp16 compute is 20 % faster at no cost in answers (A2 in docs/results.md).

| Detector (whole photo in, bird boxes out) | File | MPS ms per photo, median / p90 | CPU ms per photo |
|---|---|---|---|
| OWLv2 | 620 MB | 132 / 136 | 513 |
| OWLv2 int8 | 156 MB | 133 / 138 | 519 |
| RT-DETRv2 r50 | 172 MB | 37 / 42 | 123 |
| RT-DETRv2 r18 | 81 MB | 20 / 33 | 69 |
| D-FINE small | 42 MB | 20 / 30 | 143 |
| D-FINE nano | 15 MB | 18 / 30 | 114 |
| D-FINE nano, Core ML on the Mac (GPU) | 15.5 MB | 4 / 4 | 17 |
| YOLOv10 m / s / n (onnxruntime) | 62 / 29 / 9 MB | - | 54 / 24 / 14 |

### Data

The evaluation set is gardener's: 1,571 research-grade iNaturalist photos (1,046 birds of 42 species, 525 mammals
of 22), up to 25 per species, almost all from California and taken in September 2026, 500 px, licences CC BY-NC
1,402 / CC BY 120 / CC0 49; the list with licences, attribution and sha256 is `eval/groundtruth.csv`, and
`eval/fetch_photos.py` downloads the photos (not redistributed here). Box accuracy uses CUB-200-2011 (not
redistributed). Two wider sets are listed the same way: worldwide (`eval/inat-global.csv`, 2,632 photos of 962
species from 138 countries) and US/Canada/China (`eval/inat2026.csv`, 637 photos of 228 species), all taken in 2026,
species chosen at random. Details, biases and the
training data: [docs/data.md](docs/data.md).

### Which to pick

- **Same answers, smaller file:** the int8 BioCLIP image tower (0.63 GB) named the bird as the original did on
  1,045 of 1,046 bird photos, with the same top-1 and top-5 rates.
- **Recommended pipeline:** RT-DETRv2 r50 with OWLv2 (int8, no text tower) as the fallback, and the int8 tower in
  fp16: worldwide 99.6 % found, 77.7 % / 92.4 % against the original's 99.3 %, 77.2 % / 92.3 %, identify median
  113 ms against 205 ms (docs/results.md C1). Models on disk: 172 + 92 + 633 MB + the 22 MB names matrix, against
  620 + 3,945 MB; the SigLIP2 gate of gardener's pipeline (1.5 GB) is unchanged in both.
- **Highest bird-finding rate, and fast:** RT-DETRv2 r50 with OWLv2 as the fallback (99.5 % found, top-1 92.5 %;
  OWLv2 ran on 14 of 1,046 bird photos), or D-FINE nano with the fallback (99.3 %, 92.3 %; 15 MB first stage).
- **One detector only, most birds found:** OWLv2 or its int8 file (99.0 %); int8 is 156 MB instead of 620 MB
  (92 MB without the text tower), with top-1 92.2 % instead of 92.4 %.
- **Smallest detector with a permissive licence:** D-FINE nano (15 MB, Apache-2.0): 97.9 % found, top-1 91.1 %;
  on CUB's true boxes 98.3 % of its boxes overlap the truth by IoU 0.5 or more; 18 ms a photo on the Mac's GPU
  against OWLv2's 132 ms. It finds birds only.
- **Avoid for this use:** the YOLOv10 ONNX models (AGPL-3.0) found 82.5-91.2 % of the birds.
- The distilled students are not good enough to replace BioCLIP (60.0 % top-1 at best).

## Quick start

```
pip install torch torchvision open_clip_torch safetensors numpy pillow
# download bioclip-2.5-vith14-visual-int8.safetensors and bioclip-2.5-vith14-avilist-2026-names.npz from Releases
python examples/identify.py crop.jpg \
    --model bioclip-2.5-vith14-visual-int8.safetensors --names bioclip-2.5-vith14-avilist-2026-names.npz
```

```
crop.jpg
  1.000  Buteo lineatus (Red-shouldered Hawk)
  0.000  Buteo jamaicensis (Red-tailed Hawk)
  ...
```

The input is a photo cropped around one bird; the output is the image-only top five (no location prior).
Add `--parts parts.safetensors --flight flight.safetensors` from bioclip-bird-heads for beak/eye points and
P(flying). The example needs no gardener code, no original checkpoint and no network (tested with open_clip
3.3.0; it uses open_clip's internal `_build_vision_tower`). On a GPU it computes in fp16 since 2026-11-18;
the following check was made before that, in fp32 (two crops re-run in fp16 gave the same output). Checked on 40 bird crops of
the evaluation set against gardener's own path with the same int8 tower: the same five species in the same order
for 38, the two others differing only in the order of tied names (a pair sharing one vector, and names at
probability 0.000); probabilities within 0.001. Its head outputs against the training code's own path (float16
tower) on the same crops: 117 of 120 points within a few pixels (median 0.4 px), 3 jumped to another candidate
patch; P(flying) within 0.023, same flying/not call on all 40.

## Development scripts and gardener

Everything except `examples/` and `eval/fetch_photos.py` is the development harness and depends on
[gardener](https://github.com/lincleejun/gardener) checked out next to this repository as `../bioscan`:
`./run SCRIPT` runs a script in gardener's uv environment (its torch, open_clip and transformers versions, with
`HF_HUB_OFFLINE=1`), and the evaluation goes through gardener's engine and `gardener-engine eval`.

| File | Does |
|---|---|
| `distill/slim.py` | Writes the fp16/int8 image tower; `load()` reads it back; the block-token helper the heads use |
| `distill/detectors.py` | Other detectors in OWLv2's seat; writes and loads the OWLv2 int8 file |
| `distill/serve.py` | gardener's engine with the species encoder swapped (`--slim`, `--student`), recorded (`--record`) or the detector swapped (`--detector`) |
| `eval_teacher.sh`, `eval_student.sh`, `detbench.sh` | One evaluation run each (see docs/results.md) |
| `distill/student.py`, `train.py`, `data.py`, `embed.py`, `bench.py` | Distillation: students, teacher vectors, speed |
| `distill/fetch_inat.py`, `quality.py`, `crops.sh` | Training photos by region from iNaturalist and their checks |
| `distill/aesthetic.py`, `distill/rarity.py` | Side experiments: aesthetic score from BioCLIP tokens; rarity table |
| `eval/agree.py` | Photo-by-photo agreement and box agreement, from the published predictions (no model run) |
| `eval/crops.py`, `eval/cub_boxes.py`, `eval/speed.py` | Species model alone on crops; detector boxes on CUB; speed |
| `eval/` | Evaluation list, download script, results of every run |
| `notes/` | Research notes (Chinese) |

## License

Code: MIT ([LICENSE](LICENSE)). Released weights and data keep their upstream licences: BioCLIP 2.5 MIT,
OWLv2 Apache-2.0, AviList CC BY 4.0, TreeOfLife-200M vectors CC0, evaluation photos each under its own
Creative Commons licence (mostly CC BY-NC). Details in [MODELS.md](MODELS.md). gardener's location prior
(BirdNET) used in the evaluation is CC BY-NC-SA 4.0.
