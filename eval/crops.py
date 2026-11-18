"""The species model alone, without detector, gate or location prior: bird crops in, a name out of 11,172.

  ./run eval/crops.py > eval/results/crops/report.md        # development environment (gardener's, see README)

Two sets of crops:
1. golden: the 1,046 bird photos of eval/groundtruth.csv, each cut around the best box of the original pipeline's run
   (eval/results/teacher) with gardener's own framing (rules.crop_with_context). The same crop for every model,
   so differences are the model's alone. Truth names renamed with gardener's synonyms (eval/agree.py).
2. recorded: the 3,171 crops (birds and mammals) whose vectors the original model recorded during its pipeline run
   (data/rec/golden; the crops were saved as JPEG, shrunk to 448 px, so even the original does not reproduce its
   own vectors exactly - its row is the floor).
Models: BioCLIP 2.5 original (both towers, fp32), the fp16 and int8 image towers (fp32 and fp16 compute), and
BioCLIP 1 (imageomics/bioclip, ViT-B/16) with its own text tower encoding the same 11,172 names.
Writes per-crop results to eval/results/crops/preds.ndjson.gz.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from agree import RES, norm, preds, truth  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
NAMES = ROOT.parent / "bioscan" / "data" / "names" / "cache" / "bioclip-2.5-vith14-a4f7858b14c15128.npz"
PHOTOS = ROOT.parent / "bioscan" / "data" / "inat"


def golden_crops() -> tuple[list[Image.Image], list[str]]:
    from gardener.service.rules import crop_with_context
    gt, ref = truth(), preds("teacher")
    crops, names = [], []
    for p, r in gt.items():
        if r["kind"] != "bird" or not ref[p]["best"]:
            continue
        im = Image.open(PHOTOS / p.removeprefix("photos/")).convert("RGB")
        x0, y0, x1, y1 = ref[p]["best"]["xyxy"]
        crops.append(crop_with_context(im, (x0 * im.width, y0 * im.height, x1 * im.width, y1 * im.height)))
        names.append(norm(r["scientific"]))
    return crops, names


def encoders():
    """(name, encode(images) -> (n, d) unit vectors, logit scale, names matrix) for each model, loaded one at a time."""
    import open_clip
    from gardener.service.adapters.bioclip import BioCLIP

    from distill import slim
    z = np.load(NAMES)
    m25 = torch.from_numpy(z["matrix"]).to(DEV)
    bc = BioCLIP(DEV)
    pre = bc.preprocess

    def enc(model, half=False):
        def f(images):
            out = []
            for i in range(0, len(images), 32):
                x = torch.stack([pre(im) for im in images[i:i + 32]]).to(DEV)
                with torch.no_grad():
                    v = model(x.half() if half else x).float()
                out.append(v / v.norm(dim=-1, keepdim=True))
            return torch.cat(out)
        return f
    yield "BioCLIP 2.5 original (fp32)", enc(bc.model.visual), bc.logit_scale, m25
    for file, label in (("fp16", "fp16 file"), ("int8", "int8 file")):
        bc.model.visual.float().load_state_dict(slim.load(str(ROOT / f"models/bioclip-2.5-vith14-visual-{file}.safetensors")))
        yield f"BioCLIP 2.5 {label}, fp32 compute", enc(bc.model.visual), bc.logit_scale, m25
        bc.model.visual.half()
        yield f"BioCLIP 2.5 {label}, fp16 compute", enc(bc.model.visual, True), bc.logit_scale, m25
    del bc
    from huggingface_hub import snapshot_download
    local = "local-dir:" + snapshot_download("imageomics/bioclip")
    model, _, pre1 = open_clip.create_model_and_transforms(local, device=DEV)
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-16")              # BioCLIP 1 uses CLIP's tokenizer; open_clip ships its vocabulary
    texts = [f"a photo of {' '.join(list(t[:6]) + [t[6].split(' ')[-1]])} with common name {c}." for t, c in zip(z["taxonomy"], z["common"])]
    with torch.no_grad():
        m1 = torch.cat([torch.nn.functional.normalize(model.encode_text(tok(texts[i:i + 512]).to(DEV)).float(), dim=-1) for i in range(0, len(texts), 512)])
    pre = pre1
    yield "BioCLIP 1 (ViT-B/16), own text tower", enc(model.visual), float(model.logit_scale.exp()), m1


def main() -> None:
    z = np.load(NAMES)
    sci = [norm(s) for s in z["scientific"]]
    crops, truth_names = golden_crops()
    rec_files = sorted((ROOT / "data/rec/golden").glob("*.npy"))
    rec_vecs = np.concatenate([np.load(f).astype(np.float32) for f in rec_files])
    rec_crops = [Image.open(f.with_name(f"{f.stem}_{i:02d}.jpg")).convert("RGB") for f in rec_files for i in range(len(np.load(f)))]
    rec_first = (rec_vecs @ z["matrix"].T).argmax(1)
    rows, base = [], {}
    (RES / "crops").mkdir(exist_ok=True)
    out = gzip.open(RES / "crops" / "preds.ndjson.gz", "wt")
    for name, encode, scale, matrix in encoders():
        v = encode(crops)
        p = (scale * v @ matrix.T).softmax(-1)
        top = p.topk(5).indices.cpu().numpy()
        first = [sci[t[0]] for t in top]
        t1 = np.mean([f == t for f, t in zip(first, truth_names)])
        t5 = np.mean([t in [sci[j] for j in row] for row, t in zip(top, truth_names)])
        row = {"model": name, "top1": t1, "top5": t5}
        if matrix.shape[1] == 1024:
            vn = v.cpu().numpy()
            if not base:
                base = {"v": vn, "first": first}
            cos = (vn * base["v"]).sum(1)
            row |= {"cos_mean": cos.mean(), "cos_min": cos.min(), "same_first": sum(a == b for a, b in zip(first, base["first"]))}
            r = encode(rec_crops).cpu().numpy()
            rc = (r * rec_vecs).sum(1)
            row |= {"rec_cos_mean": rc.mean(), "rec_cos_min": rc.min(), "rec_same_first": int(((r @ z["matrix"].T).argmax(1) == rec_first).sum())}
        rows.append(row)
        for c, f5 in zip(truth_names, top):
            out.write(json.dumps({"model": name, "truth": c, "top5": [sci[j] for j in f5]}) + "\n")
        print(f"done: {name}", file=sys.stderr, flush=True)
    out.close()
    n = len(crops)
    print(f"# Species model alone on bird crops\n\n{n} bird crops of the evaluation set (the original pipeline's best box on each of the 1,046 bird photos "
          f"that got one), 11,172 AviList names, image only (no location prior). Device: {DEV}.\n")
    print("| model | top-1 | top-5 | cosine to the original's vector: mean / min | same first name as the original |")
    print("|---|---|---|---|---|")
    for r in rows:
        extra = f"{r['cos_mean']:.5f} / {r['cos_min']:.5f} | {r['same_first']} of {n}" if "cos_mean" in r else "- (other vector space) | -"
        print(f"| {r['model']} | {100 * r['top1']:.1f} % | {100 * r['top5']:.1f} % | {extra} |")
    print(f"\n## Against the vectors the original recorded in its pipeline run ({len(rec_crops)} crops, birds and mammals, saved as JPEG)\n")
    print("| model | cosine: mean / min | same first bird name (of the 11,172) |")
    print("|---|---|---|")
    for r in rows:
        if "rec_cos_mean" in r:
            print(f"| {r['model']} | {r['rec_cos_mean']:.5f} / {r['rec_cos_min']:.5f} | {r['rec_same_first']} of {len(rec_crops)} |")


if __name__ == "__main__":
    main()
