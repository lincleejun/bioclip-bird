# bioclip-bird

Smaller models for naming birds in photos, measured end to end on 1,571 iNaturalist photos.

> 中文简介：把 [gardener](https://github.com/lincleejun/gardener) 识鸟用的 BioCLIP 2.5（3.9 GB）和 OWLv2 检测器（620 MB）
> 压小：BioCLIP 只留看图的一半并存成 8 位（0.63 GB），OWLv2 存成 8 位（156 MB），评测集上成绩不变。
> 附 11,172 种鸟的名录矩阵、不依赖 gardener 的最小推理示例、全部评测清单和结果。蒸馏出的小学生模型离教师差很远，只公开结果不发权重。
> 鸟眼鸟嘴和飞行两个小头在另一个仓库（权重仅限非商用）。

## What is here

1. **BioCLIP 2.5 image tower, int8** (0.63 GB, from 3.9 GB) and **fp16** (1.26 GB): the upstream weights
   without the text tower, rounded. Birds on the evaluation set: identical to the original (92.4 % / 96.9 %);
   mammals 91.6 % / 95.4 % against 91.4 % / 95.4 %.
2. **Bird names matrix**: 11,172 AviList v2026 species as BioCLIP 2.5 text vectors (22 MB), plus the logit
   scale (98.86). With the image tower it names a bird; no text tower needed.
3. **OWLv2 detector, int8** (156 MB, from 620 MB): same bird detection rate on the evaluation set.
4. **Evaluation**: the 1,571-photo list with licences and a download script, every run's report and
   per-photo predictions, and the results write-up.

Weights are attached to the [GitHub Releases](https://github.com/lincleejun/bioclip-bird/releases), not in git.
File list, sizes, sha256, upstream models and licences: [MODELS.md](MODELS.md).

Add-on: **beak/eye points and flying-or-not** heads on the same tower live in
[bioclip-bird-heads](https://github.com/lincleejun/bioclip-bird-heads); their weights are for non-commercial use
only (trained on CUB-200-2011 and mostly CC BY-NC photos).

## Results (birds, n = 1,046; full tables in [docs/results.md](docs/results.md))

Whole gardener pipeline (SigLIP2 gate, detector, BioCLIP, BirdNET location prior); only the named part changes.

| Variant | Size | Bird found | Top-1 | Top-5 |
|---|---|---|---|---|
| BioCLIP 2.5 original + OWLv2 (gardener today) | 3.9 GB + 620 MB | 99.0 % | 92.4 % | 96.9 % |
| BioCLIP int8 image tower | 0.63 GB | 99.0 % | 92.4 % | 96.9 % |
| OWLv2 int8 | 156 MB | 99.0 % | 92.2 % | 96.9 % |
| D-FINE nano instead of OWLv2 (finds birds only) | 15 MB | 97.9 % | 91.1 % | 95.0 % |
| Distilled ViT-S/16 student in BioCLIP's place | 88 MB | 99.2 % | 60.0 % | 78.7 % |

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
3.3.0; it uses open_clip's internal `_build_vision_tower`). Checked on 40 bird crops of
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
| `eval/` | Evaluation list, download script, results of every run |
| `notes/` | Research notes (Chinese) |

## License

Code: MIT ([LICENSE](LICENSE)). Released weights and data keep their upstream licences: BioCLIP 2.5 MIT,
OWLv2 Apache-2.0, AviList CC BY 4.0, TreeOfLife-200M vectors CC0, evaluation photos each under its own
Creative Commons licence (mostly CC BY-NC). Details in [MODELS.md](MODELS.md). gardener's location prior
(BirdNET) used in the evaluation is CC BY-NC-SA 4.0.
