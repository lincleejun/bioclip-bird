"""A second, wider evaluation set: 2026 research-grade iNaturalist bird photos from the US, Canada and China.

  ./run eval/build_eval2.py            # -> eval/inat2026.csv (the list) and eval/photos2026/ (photos, not in git)

Why: the first set (eval/groundtruth.csv) is 42 common California species, almost all from September 2026. This one
samples 500 species at random (fixed seed) from iNaturalist's bird lists of the US, Canada and China
(data/inat-raw/species.csv, 2,179 species), so common and uncommon species, three countries and every month of 2026
so far. For each species, up to 3 research-grade observations made in 2026 in a country whose list has it, each by
a different observer, CC0 / CC BY / CC BY-NC photos only, one photo per observation at "medium" size (500 px, as the
first set).

Kept out: every observation and photo the students trained on (data/inat-raw/manifest.csv) and the first set's, and
any observation by the same person of the same species on the same day as a training photo. Photos from 2026 are
newer than BioCLIP 2.5's training data (TreeOfLife-200M), so the species model has not seen them either.
Resumable: listed species and downloaded files are skipped. One API call or photo a second.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from distill.fetch_inat import LICENSES, Api, get, golden_ids  # noqa: E402

PLACES = {"US": 1, "CA": 6712, "CN": 6903}
N_SPECIES, PER_SPECIES, SEED, SINCE = 500, 3, 2026, "2026-01-01"
OUT = ROOT / "eval" / "inat2026.csv"
PHOTOS = ROOT / "eval" / "photos2026"
LISTED = ROOT / "data" / "eval2-listed.jsonl"           # API answers kept, so a rerun lists nothing twice
FIELDS = ["path", "scientific", "tier", "lat", "lon", "taken_at", "source", "kind", "license", "attribution",
          "country", "observer", "sha256"]


def training() -> tuple[set[str], set[tuple[str, str, str]]]:
    """Observation and photo ids trained on (and the first set's), and (photographer, species, day) of training photos."""
    ids, same_day = golden_ids(), set()
    with open(ROOT / "data" / "inat-raw" / "manifest.csv", newline="") as f:
        for r in csv.DictReader(f):
            ids |= {r["photo_id"], r["observation_id"]}
            who = r["attribution"].removeprefix("(c) ").split(", some rights")[0].split(", no rights")[0]
            same_day.add((who, r["scientific"], r["observed_on"]))
    return ids, same_day


def pick() -> list[tuple[str, str, str]]:
    """(taxon_id, scientific, country) for N_SPECIES species; the country is one whose list has the species."""
    with open(ROOT / "data" / "inat-raw" / "species.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(SEED)
    out = []
    for r in rng.sample(rows, N_SPECIES):
        countries = [c.split(":")[0] for c in r["regions"].split(";")]
        out.append((r["taxon_id"], r["scientific"], rng.choice(countries)))
    return out


def main() -> None:
    api, (seen_ids, same_day) = Api(), training()
    listed = {}
    if LISTED.exists():
        listed = {(d["taxon_id"], d.get("country")): d for d in map(json.loads, LISTED.open())}
    rows, species = [], pick()
    for n, (tid, name, country) in enumerate(species, 1):
        if (tid, country) not in listed:
            ans = api("observations", taxon_id=tid, place_id=PLACES[country], quality_grade="research", photos="true",
                      photo_license=",".join(LICENSES), d1=SINCE, per_page=60, order_by="observed_on", order="desc")
            listed[tid, country] = {"taxon_id": tid, "country": country, "results": ans["results"]}
            with LISTED.open("a") as f:
                f.write(json.dumps(listed[tid, country]) + "\n")
        users = set()
        for o in listed[tid, country]["results"]:
            if len(users) == PER_SPECIES:
                break
            photo = next((p for p in o.get("photos") or [] if p.get("license_code") in LICENSES), None)
            user = (o.get("user") or {}).get("login", "")
            who = (photo or {}).get("attribution", "").removeprefix("(c) ").split(", some rights")[0].split(", no rights")[0]
            if (not photo or not user or user in users or not o.get("location") or str(o["id"]) in seen_ids
                    or str(photo["id"]) in seen_ids or (who, name, o.get("observed_on")) in same_day
                    or (o.get("taxon") or {}).get("rank") != "species"):
                continue
            users.add(user)
            lat, lon = o["location"].split(",")
            rows.append({"path": f"photos2026/{o['taxon']['name'].replace(' ', '_')}/{o['id']}_{photo['id']}.jpg",
                         "scientific": o["taxon"]["name"], "tier": "inat2026", "lat": lat, "lon": lon,
                         "taken_at": o.get("time_observed_at") or o.get("observed_on"), "kind": "bird",
                         "source": f"https://www.inaturalist.org/observations/{o['id']}", "license": photo["license_code"],
                         "attribution": photo["attribution"], "country": country, "observer": user,
                         "url": photo["url"].replace("/square.", "/medium.")})
        if n % 50 == 0:
            print(f"listed {n}/{len(species)} species, {len(rows)} photos", flush=True)
    for n, r in enumerate(rows, 1):
        dest = ROOT / "eval" / r["path"]
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            time.sleep(max(0.0, 1.0 - (time.monotonic() - api.last)))     # photos share the one-a-second pace
            api.last = time.monotonic()
            dest.write_bytes(get(r.pop("url"), tries=3))
        else:
            r.pop("url")
        r["sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
        if n % 200 == 0:
            print(f"photos {n}/{len(rows)}", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(rows)} photos of {len({r['scientific'] for r in rows})} species -> {OUT}")


if __name__ == "__main__":
    main()
