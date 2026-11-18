"""How often each bird is recorded where and when (GBIF occurrence counts) and how threatened it is (IUCN category).

  ./run distill/rarity.py --countries US CA CN      # -> data/rarity/counts.csv, species.csv, regions.csv (resumable)

counts.csv: region, month, species_key, count: GBIF occurrence records of class Aves per species, for each country
and each of its first-level divisions (GADM ids such as USA.5_1 = California), per calendar month.
species.csv: species_key, GBIF name, the matching name in gardener's bird list (AviList; empty when none), IUCN code.
Only public aggregate queries: about 12 calls per region plus two per species seen, two a second.
A count is how often a bird was *reported*, not how many there are; compare within one region, never across regions.
Cite GBIF.org (the occurrence API, date of the run) and the IUCN Red List when these tables leave this machine.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from distill.data import BIRDS, NAMES

API, UA, AVES = "https://api.gbif.org/v1/", "bird-photo-fetch/0.1 (personal research)", 212
OUT = Path(__file__).resolve().parents[1] / "data" / "rarity"


def get(path: str, **q):
    url = API + path + ("?" + urllib.parse.urlencode(q) if q else "")
    for k in range(5):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=180) as r:
                body = r.read()
                return json.loads(body) if body else None
        except (urllib.error.URLError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                return None
            time.sleep(30 * (k + 1))
    raise SystemExit(f"gave up on {url}")


def facet(field: str, **q) -> list[dict]:
    return get("occurrence/search", taxonKey=AVES, facet=field, facetLimit=30000, limit=0, **q)["facets"][0]["counts"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+", required=True, help="ISO 3166 two-letter codes")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if (OUT / "counts.csv").exists():
        with open(OUT / "counts.csv", newline="") as f:
            done = {(r["region"], r["month"]) for r in csv.DictReader(f)}
    else:
        (OUT / "counts.csv").write_text("region,month,species_key,count\n")
    regions = []
    for c in a.countries:
        regions.append((c, {"country": c}))
        regions += [(g["name"], {"gadmGid": g["name"]}) for g in facet("gadmGid", country=c) if re.fullmatch(r"[A-Z]{3}\.\d+_\d+", g["name"])]
    (OUT / "regions.csv").write_text("region\n" + "\n".join(r for r, _ in regions) + "\n")
    print(f"{len(regions)} regions", flush=True)
    for n, (region, q) in enumerate(regions):
        for month in range(1, 13):
            if (region, str(month)) in done:
                continue
            rows = facet("speciesKey", month=month, **q)
            with open(OUT / "counts.csv", "a") as f:
                f.writelines(f"{region},{month},{r['name']},{r['count']}\n" for r in rows)
                if not rows:
                    f.write(f"{region},{month},,0\n")                 # so a re-run knows the month was asked
        if n % 10 == 0:
            print(f"{n + 1}/{len(regions)} regions", flush=True)
    with open(OUT / "counts.csv", newline="") as f:
        keys = sorted({r["species_key"] for r in csv.DictReader(f) if r["species_key"]}, key=int)
    known = {}
    if (OUT / "species.csv").exists():
        with open(OUT / "species.csv", newline="") as f:
            known = {r["species_key"]: r for r in csv.DictReader(f)}
    ours = set(np.load(NAMES / BIRDS, allow_pickle=False)["scientific"].tolist())
    with open(OUT / "species.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["species_key", "gbif_name", "avilist_name", "iucn"])
        w.writeheader()
        for n, k in enumerate(keys):
            if k not in known:
                name = (get(f"species/{k}") or {}).get("canonicalName", "")
                known[k] = {"species_key": k, "gbif_name": name, "avilist_name": name if name in ours else "",
                            "iucn": (get(f"species/{k}/iucnRedListCategory") or {}).get("code", "")}
            w.writerow(known[k])
            f.flush()
            if n % 500 == 0:
                print(f"{n}/{len(keys)} species", flush=True)
    hit = sum(bool(r["avilist_name"]) for r in known.values())
    print(f"done: {len(keys)} species keys, {hit} with the same name in gardener's list ({hit / len(keys):.1%}), {sum(bool(r['iucn']) for r in known.values())} with an IUCN category")


if __name__ == "__main__":
    main()
