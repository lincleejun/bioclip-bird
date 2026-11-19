"""Location priors side by side on the same bird crops: none, BirdNET geo (gardener's), and GBIF record counts by region
and month (data/rarity, from distill/rarity.py) as a soft prior or a hard list.

  ./run eval/priors.py groundtruth.csv e1-teacher > eval/results/priors-e1.md    # first set; boxes of that run
  ./run eval/priors.py inat2026.csv e2-teacher > eval/results/priors-e2.md       # second set

Crops: the run's best box on each bird photo, gardener's framing; species model: the int8 image tower, fp16 compute;
p_visual = softmax over the 11,172 AviList names. Top-1 ties (names sharing one vector) go to the lower row, as an
argmax does. Region of a photo: its state/province (GADM level 1 in the GBIF tables) and else its country. The first
set's bird photos are all in California; the second set's come from each observation's iNaturalist places (cached
in data/eval2-places.json; a few API calls the first time).
GBIF soft prior: logit += beta * log((n + 1) / (N + S)), n = the species' records in the region in the photo's month
+-1, N their total, S the number of names. Hard list: names with n > 0 only. GBIF names are matched to AviList by
name, by GBIF synonyms (data/rarity/synonyms.csv), by gardener's synonyms, and by an epithet unique in AviList.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from agree import RES, SYN, norm, preds  # noqa: E402

NAMES = ROOT.parent / "bioscan" / "data" / "names" / "cache" / "bioclip-2.5-vith14-a4f7858b14c15128.npz"
RARITY = ROOT / "data" / "rarity"
CA_BOX = (32.4, 42.1, -124.6, -114.1)
BETAS = (0.5, 1.0)


def truth(list_name: str) -> dict[str, dict]:
    alias = {norm(r["alias"]): r["avilist_scientific"].strip() for r in csv.DictReader(open(SYN, newline=""))
             if r["source"].strip() in ("inat", "spelling")}
    rows = csv.DictReader(open(HERE / list_name, newline=""))
    return {r["path"]: {**r, "scientific": alias.get(norm(r["scientific"]), r["scientific"])} for r in rows if r["kind"] == "bird"}


def gbif_to_avilist(sci: list[str]) -> dict[str, int]:
    """GBIF species key -> row of the names matrix."""
    idx = {s: i for i, s in enumerate(sci)}
    alias = {norm(r["alias"]): norm(r["avilist_scientific"]) for r in csv.DictReader(open(SYN, newline=""))}
    epithet = defaultdict(list)
    for s in sci:
        epithet[s.split()[-1]].append(s)
    syn = {r["species_key"]: norm(r["avilist_name"]) for r in csv.DictReader(open(RARITY / "synonyms.csv")) if r["avilist_name"]}
    out = {}
    for r in csv.DictReader(open(RARITY / "species.csv")):
        k, g, a = r["species_key"], norm(r["gbif_name"]), norm(r["avilist_name"])
        name = alias.get(a) if a and alias.get(a) in idx else a or syn.get(k) or (alias.get(g) if alias.get(g) in idx else None)
        if not name and len(g.split()) == 2 and len(epithet[g.split()[1]]) == 1:
            name = epithet[g.split()[1]][0]
        if name in idx:
            out[k] = idx[name]
    return out


def counts(regions: set[str], key2row: dict[str, int], n_names: int) -> dict[str, np.ndarray]:
    """region -> (13, names) GBIF records per month (row 0 unused)."""
    out, seen = {r: np.zeros((13, n_names)) for r in regions}, set()
    for r in csv.DictReader(open(RARITY / "counts.csv")):
        if r["region"] in out and r["species_key"] in key2row:
            k = (r["region"], r["month"], r["species_key"])
            if k not in seen:                                          # the file repeats some blocks
                seen.add(k)
                out[r["region"]][int(r["month"]), key2row[r["species_key"]]] += int(r["count"])
    return out


def regions_of(rows: dict[str, dict]) -> dict[str, str]:
    """photo -> GADM region code (state/province when it can be named, else the country)."""
    names = {r["name_en"].lower(): r["code"] for r in csv.DictReader(open(RARITY / "region_names.csv"))}
    out = {}
    if all("country" not in r for r in rows.values()):                 # first set: California by its coordinates
        for p, r in rows.items():
            lat, lon = float(r["lat"]), float(r["lon"])
            out[p] = "USA.5_1" if CA_BOX[0] < lat < CA_BOX[1] and CA_BOX[2] < lon < CA_BOX[3] else "US"
        return out
    from distill.fetch_inat import Api
    obs_places = {}
    for line in open(ROOT / "data" / "eval2-listed.jsonl"):
        for o in json.loads(line)["results"]:
            obs_places[str(o["id"])] = o.get("place_ids") or []
    cache = ROOT / "data" / "eval2-places.json"
    places = json.loads(cache.read_text()) if cache.exists() else {}
    need = sorted({str(i) for p in rows for i in obs_places.get(Path(p).stem.split("_")[0], [])} - set(places))
    api = Api()
    for i in range(0, len(need), 100):
        for pl in api(f"places/{','.join(need[i:i + 100])}")["results"]:
            places[str(pl["id"])] = {"name": pl["name"], "admin_level": pl.get("admin_level")}
        cache.write_text(json.dumps(places))
    for p, r in rows.items():
        state = next((places[str(i)]["name"] for i in obs_places.get(Path(p).stem.split("_")[0], [])
                      if places.get(str(i), {}).get("admin_level") == 10), None)
        out[p] = names.get((state or "").lower(), r["country"])
    return out


def main() -> None:
    from gardener.service.adapters.geo import GeoPrior, priors_for
    from gardener.service.engine import load_species
    from gardener.service.rules import crop_with_context
    from PIL import Image

    from distill import slim
    list_name, run = sys.argv[1], sys.argv[2]
    rows, ref = truth(list_name), preds(run)
    keep = [p for p in rows if ref.get(p, {}).get("best")]
    z = np.load(NAMES)
    sci = [norm(s) for s in z["scientific"]]
    region = regions_of({p: rows[p] for p in keep})

    crops = []
    for p in keep:
        f = HERE / p if (HERE / p).exists() else ROOT.parent / "bioscan" / "data" / "inat" / p.removeprefix("photos/")
        im = Image.open(f).convert("RGB")
        x0, y0, x1, y1 = ref[p]["best"]["xyxy"]
        crops.append(crop_with_context(im, (x0 * im.width, y0 * im.height, x1 * im.width, y1 * im.height)))
    _, lists = load_species("cpu")                                        # gardener's name lists, for the BirdNET prior
    tower = slim.build(str(ROOT / "models/bioclip-2.5-vith14-visual-int8.safetensors"), "mps", half=True)
    from torchvision import transforms
    prep = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224),
                               transforms.ToTensor(), transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    vs = []
    with torch.no_grad():
        for i in range(0, len(crops), 32):
            v = tower(torch.stack([prep(c) for c in crops[i:i + 32]]).to("mps").half()).float()
            vs.append((v / v.norm(dim=-1, keepdim=True)).cpu().numpy())
    logits = 98.8645 * np.concatenate(vs) @ z["matrix"].T.astype(np.float32)
    p_vis = np.exp(logits - logits.max(1, keepdims=True))
    p_vis /= p_vis.sum(1, keepdims=True)

    bird_prior = priors_for(lists, GeoPrior.load())["bird"]
    birdnet = np.stack([bird_prior.posterior(pv, bird_prior.p_geo(float(rows[p]["lat"]), float(rows[p]["lon"]), rows[p]["taken_at"]))
                        for p, pv in zip(keep, p_vis)])
    key2row = gbif_to_avilist(sci)
    n = counts(set(region.values()), key2row, len(sci))
    month = [int(rows[p]["taken_at"][5:7]) for p in keep]
    win = np.stack([n[region[p]][[(m - 2 + d) % 12 + 1 for d in range(3)]].sum(0) for p, m in zip(keep, month)])
    share = np.log((win + 1) / (win.sum(1, keepdims=True) + win.shape[1]))
    truths = [norm(rows[p]["scientific"]) for p in keep]

    def top5(score):
        return np.argsort(-score, 1, kind="stable")[:, :5]

    variants = {"no location": top5(logits), "BirdNET geo (gardener)": top5(np.log(birdnet + 1e-30))}
    for b in BETAS:
        variants[f"GBIF soft, beta {b}"] = top5(logits + b * share)
    variants["GBIF hard list"] = top5(np.where(win > 0, logits, -1e9))

    groups = {"all": [True] * len(keep)}
    if "country" in rows[keep[0]]:
        for c in sorted({rows[p]["country"] for p in keep}):
            groups[f"country {c}"] = [rows[p]["country"] == c for p in keep]
        obs = {norm(r["scientific"]): sum(int(x.split(":")[1]) for x in r["regions"].split())
               for r in csv.DictReader(open(ROOT / "data" / "inat-raw" / "species.csv"))}
        cut = np.quantile([obs.get(norm(rows[p]["scientific"]), 0) for p in keep], [1 / 3, 2 / 3])
        for label, lo, hi in (("least observed third", -1, cut[0]), ("middle third", cut[0], cut[1]), ("most observed third", cut[1], 1e18)):
            groups[f"species: {label}"] = [lo < obs.get(norm(rows[p]["scientific"]), 0) <= hi for p in keep]

    print(f"# Location priors on the same bird crops ({list_name}, boxes of run `{run}`)\n")
    print(f"{len(keep)} bird crops of {len(rows)} bird photos ({len(rows) - len(keep)} without a box are left out), "
          f"{len(set(truths))} species. Regions: {dict(Counter(region.values()).most_common(8))}"
          f"{' ...' if len(set(region.values())) > 8 else ''}. Truth names on no GBIF list of their region (any month): "
          f"{sum(n[region[p]][1:].sum(0)[sci.index(t)] == 0 for p, t in zip(keep, truths) if t in sci)}.\n")
    print("| prior | " + " | ".join(f"{g} (n = {sum(m)}) top-1 / top-5" for g, m in groups.items()) + " |")
    print("|---|" + "---|" * len(groups))
    for name, top in variants.items():
        cells = []
        for m in groups.values():
            ix = [i for i, k in enumerate(m) if k]
            t1 = np.mean([sci[top[i, 0]] == truths[i] for i in ix])
            t5 = np.mean([truths[i] in [sci[j] for j in top[i]] for i in ix])
            cells.append(f"{100 * t1:.1f} % / {100 * t5:.1f} %")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
