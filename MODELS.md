# Released files

Attached to [GitHub Releases](https://github.com/lincleejun/bioclip-bird/releases) with a `SHA256SUMS` file;
not in git. Each file keeps the licence of what it was made from: the MIT licence of this repository covers
the code, not these upstream works.

| File | Size (bytes) | sha256 |
|---|---|---|
| `bioclip-2.5-vith14-visual-int8.safetensors` | 633,415,546 | `126ea3288dcc15a474df3f931b64e2524cad4967bb6c1f7e0ad30009e0eacfe5` |
| `bioclip-2.5-vith14-visual-fp16.safetensors` | 1,264,197,296 | `b629981cfa40eee16ff831c9516b00181ecb105e8e6c5c64c12df9ded78549ae` |
| `bioclip-2.5-vith14-avilist-2026-names.npz` | 21,597,460 | `c69f298195f72d14747145bb92b92b8d98a71866066f644965d936b1ea679e5f` |
| `owlv2-base-patch16-ensemble-int8.safetensors` | 155,652,122 | `e07efa451b90aece527cb4e78ac1761885de6bc41d4c0d35b171da9301ad5679` |
| `owlv2-base-patch16-ensemble-vision-int8.safetensors` (release v0.2) | 91,971,770 | `1f4231618f4f5c5f7029d3e0694370ef07429af5342cd2eb9cbec1d6d72914a7` |

## BioCLIP 2.5 image tower, int8 and fp16

- **Upstream:** [`imageomics/bioclip-2.5-vith14`](https://huggingface.co/imageomics/bioclip-2.5-vith14)
  (BioCLIP 2.5 Huge, ViT-H/14), revision `6e3d04e3d6522012c88181085c5ae666e14c45cd`. **Licence: MIT.**
  Please cite BioCLIP 2.5 as its model card asks.
- **What we changed:** kept the image tower only (the text tower is dropped: species names come precomputed in
  the names file). fp16: every tensor stored as float16 (3.94 GB checkpoint -> 1.26 GB). int8: every weight
  matrix with more than 100,000 numbers stored as int8 with one float16 scale per row (`KEY.q8`, `KEY.s8`);
  biases, norms and embeddings float16 (0.63 GB). Written by `distill/slim.py`; read back with `slim.load()` or
  the 20-line `unpack()` in `examples/identify.py`. Compute stays float: the file is smaller, memory in use is not.
- **Training:** none. These are the upstream weights, rounded.

## Bird names matrix

- `bioclip-2.5-vith14-avilist-2026-names.npz`: one row per species of **AviList v2026** (11,172 bird species):
  `scientific`, `common`, `taxonomy` (7 ranks), `matrix` (11,172 x 1024 float16, L2-normalised text vectors),
  `tol_how`, `logit_scale` (98.8645, BioCLIP 2.5's learned scale), and the revisions it was built with.
  A copy of gardener's cache `bioclip-2.5-vith14-a4f7858b14c15128.npz` with the matrix stored as float16 and
  compressed (59 MB -> 22 MB). On the 3,171 recorded evaluation-crop vectors the first-ranked species is the same
  as with the float32 matrix for all 3,171, and the top-5 set for 3,166; cosine changes by at most 4e-5.
- **Where the vectors come from:** 9,416 rows are the official precomputed BioCLIP 2.5 text vectors of
  [`imageomics/TreeOfLife-200M`](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) (revision
  `5f2dc493b3dc0e544438a04038ab15faa646b749`): 9,373 by exact scientific name, 43 through a synonym
  (`tol_how`). The other 1,756 were encoded with the BioCLIP 2.5 text tower in TreeOfLife's own text format
  (`an image of <7 ranks> with common name <name>.`). Built by gardener (`gardener/service/names.py`).
- **Licences:** names and taxonomy from AviList, **CC BY 4.0** - cite: AviList Core Team. 2026. *AviList: The
  Global Avian Checklist, v2026.* https://doi.org/10.2173/avilist.v2026. TreeOfLife-200M text vectors: CC0.
  BioCLIP 2.5 text tower: MIT.
- Two AviList species can share one vector (e.g. *Circus cyaneus* and *C. hudsonius*, matched to the same
  TreeOfLife row); by image alone they tie, and gardener separates them with its location prior.

## OWLv2, int8

- **Upstream:** [`google/owlv2-base-patch16-ensemble`](https://huggingface.co/google/owlv2-base-patch16-ensemble),
  revision `cfd3195ba4ea9592eec887ded089f4c08eff231d`. **Licence: Apache-2.0.** This file is a modified version
  of that work: the whole model (text tower included, 63 M of 155 M numbers) stored in the int8 format above,
  620 MB -> 156 MB. Written by `./run distill/detectors.py`.
- **Loading:** `distill/detectors.py owlv2_int8()` builds the model from the upstream `config.json` and reads
  the weights from this file only; the processor (tokenizer and image settings) still comes from the upstream
  repository's small config files.
- **Training:** none.

## OWLv2, int8, without the text tower (release v0.2)

- **Upstream and licence:** as above (Apache-2.0); a modified version of that work.
- **What it is:** the int8 file above without the text tower and text projection, plus `query_embeds`, the text
  vectors of gardener's 47 detector prompts (file metadata `prompts`), worked out once by the full int8 model and
  stored as fp16. 156 MB -> 92 MB. Written by `./run distill/detectors.py --vision`.
- **Loading:** `distill/detectors.py OwlVision` (image processor settings from the upstream repository). It accepts
  only the 47 stored prompts; another prompt needs the full file.
- **Checked:** same boxes and prompts as the full int8 file on 40 bird and 20 mammal photos, box corners identical,
  confidences within 0.0001.

## Not released here

- **Beak/eye and flight heads:** in [bioclip-bird-heads](https://github.com/lincleejun/bioclip-bird-heads),
  non-commercial only (trained on CUB-200-2011 and mostly CC BY-NC iNaturalist photos).
- **Distilled students** (`distill/train.py`): far below the teacher (see [docs/results.md](docs/results.md)) and
  trained on photos licensed for personal research only.
- **Evaluation photos:** `eval/groundtruth.csv` lists the 1,571 iNaturalist photos with their licence and
  attribution (CC BY-NC 1,402, CC BY 120, CC0 49); `eval/fetch_photos.py` downloads them from iNaturalist.
