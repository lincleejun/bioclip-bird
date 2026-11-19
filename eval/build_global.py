"""A worldwide evaluation set: 2026 research-grade iNaturalist bird photos of species drawn from every bird on Earth.

  ./run eval/build_global.py            # -> eval/inat-global.csv (the list) and eval/photosglobal/ (photos, not in git)

1,200 species drawn at random (fixed seed) from every active, non-extinct bird species iNaturalist knows (about
11,400). For each, research-grade observations made anywhere in 2026 with a CC0 / CC BY / CC BY-NC photo; of the
newest 60, up to 3 picked at random, each by a different observer; one photo per observation at "medium" size
(500 px). Many species have no 2026 observation, so the set holds fewer species than drawn, leaning to the commoner
ones. Each photo's country comes from its observation's iNaturalist places (admin level 0).

Kept out, as in eval/build_eval2.py: every observation and photo the students trained on and those of the first set,
and any observation by the same person of the same species on the same day as a training photo. 2026 photos are
newer than BioCLIP 2.5's training data. Resumable (API answers cached in data/global-*.json*); one call or photo a second.
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
sys.path.insert(0, str(ROOT / "eval"))
from build_eval2 import training  # noqa: E402

from distill.fetch_inat import LICENSES, Api, get  # noqa: E402

N_SPECIES, PER_SPECIES, SEED, SINCE = 1200, 3, 2026, "2026-01-01"
OUT = ROOT / "eval" / "inat-global.csv"
TAXA = ROOT / "data" / "global-taxa.json"
LISTED = ROOT / "data" / "global-listed.jsonl"
PLACES = ROOT / "data" / "global-places.json"
FIELDS = ["path", "scientific", "tier", "lat", "lon", "taken_at", "source", "kind", "license", "attribution",
          "country", "country_name", "observer", "sha256"]


def all_species(api: Api) -> list[dict]:
    if TAXA.exists():
        return json.loads(TAXA.read_text())
    out, above = [], 0
    while True:
        page = api("taxa", taxon_id=3, rank="species", is_active="true", per_page=200, order_by="id", order="asc", id_above=above)
        if not page["results"]:
            break
        out += [{"id": t["id"], "name": t["name"], "observations": t.get("observations_count", 0)}
                for t in page["results"] if not t.get("extinct")]
        above = page["results"][-1]["id"]
    TAXA.write_text(json.dumps(out))
    return out


def slim(o: dict) -> dict:
    """What we keep of an observation (the full answers are large)."""
    return {"id": o["id"], "user": (o.get("user") or {}).get("login", ""), "observed_on": o.get("observed_on"),
            "time": o.get("time_observed_at") or o.get("observed_on"), "location": o.get("location"),
            "place_ids": o.get("place_ids") or [], "taxon": {"name": (o.get("taxon") or {}).get("name"), "rank": (o.get("taxon") or {}).get("rank")},
            "photos": [{"id": p["id"], "license_code": p.get("license_code"), "attribution": p.get("attribution"), "url": p.get("url")}
                       for p in o.get("photos") or []]}


def countries(api: Api, place_ids: set[int]) -> dict[str, dict]:
    """place id -> {name, admin_level}, cached; only ids not seen before are asked for."""
    places = json.loads(PLACES.read_text()) if PLACES.exists() else {}
    need = sorted(str(i) for i in place_ids if str(i) not in places)
    for i in range(0, len(need), 100):
        for pl in api(f"places/{','.join(need[i:i + 100])}")["results"]:
            places[str(pl["id"])] = {"name": pl["name"], "admin_level": pl.get("admin_level")}
        for k in need[i:i + 100]:
            places.setdefault(k, {"name": "", "admin_level": None})
        PLACES.write_text(json.dumps(places))
    return places


ALIASES = {"china": "china mainland", "republic of congo": "congo - brazzaville",
           "democratic republic of the congo": "congo - kinshasa", "myanmar": "myanmar (burma)"}


def plain(name: str) -> str:
    """Country names as iNaturalist and the GBIF table write them differently: one spelling for both."""
    n = name.lower().replace("&", "and").replace("’", "'").replace("saint ", "st. ").strip()
    n = n.removeprefix("the ").removeprefix("commonwealth of the ")
    return ALIASES.get(n, n)


def main() -> None:
    api, (seen_ids, same_day) = Api(), training()
    rng = random.Random(SEED)
    species = rng.sample(all_species(api), N_SPECIES)
    listed = {d["taxon_id"]: d for d in map(json.loads, LISTED.open())} if LISTED.exists() else {}
    rows = []
    for n, sp in enumerate(species, 1):
        tid = sp["id"]
        if tid not in listed:
            ans = api("observations", taxon_id=tid, quality_grade="research", photos="true", photo_license=",".join(LICENSES),
                      d1=SINCE, per_page=60, order_by="observed_on", order="desc")
            listed[tid] = {"taxon_id": tid, "results": [slim(o) for o in ans["results"]]}
            with LISTED.open("a") as f:
                f.write(json.dumps(listed[tid]) + "\n")
        results = listed[tid]["results"][:]
        random.Random(f"{SEED}-{tid}").shuffle(results)
        users = set()
        for o in results:
            if len(users) == PER_SPECIES:
                break
            photo = next((p for p in o["photos"] if p.get("license_code") in LICENSES), None)
            who = (photo or {}).get("attribution", "").removeprefix("(c) ").split(", some rights")[0].split(", no rights")[0]
            name = o["taxon"]["name"]
            if (not photo or not o["user"] or o["user"] in users or not o["location"] or str(o["id"]) in seen_ids
                    or str(photo["id"]) in seen_ids or (who, name, o["observed_on"]) in same_day or o["taxon"]["rank"] != "species"):
                continue
            users.add(o["user"])
            lat, lon = o["location"].split(",")
            rows.append({"path": f"photosglobal/{name.replace(' ', '_')}/{o['id']}_{photo['id']}.jpg", "scientific": name,
                         "tier": "inat-global", "lat": lat, "lon": lon, "taken_at": o["time"], "kind": "bird",
                         "source": f"https://www.inaturalist.org/observations/{o['id']}", "license": photo["license_code"],
                         "attribution": photo["attribution"], "observer": o["user"], "place_ids": o["place_ids"],
                         "url": photo["url"].replace("/square.", "/medium.")})
        if n % 100 == 0:
            print(f"listed {n}/{len(species)} species, {len(rows)} photos", flush=True)
    places = countries(api, {i for r in rows for i in r["place_ids"]})
    names = {plain(r["name_en"]): r["code"] for r in csv.DictReader(open(ROOT / "data/rarity-global/country_names.tsv"), delimiter="\t")}
    for r in rows:
        country = next((places[str(i)]["name"] for i in r.pop("place_ids") if places.get(str(i), {}).get("admin_level") == 0), "")
        r["country_name"], r["country"] = country, names.get(plain(country), "")
    for n, r in enumerate(rows, 1):
        dest = ROOT / "eval" / r["path"]
        url = r.pop("url")
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            time.sleep(max(0.0, 1.0 - (time.monotonic() - api.last)))     # photos share the one-a-second pace
            api.last = time.monotonic()
            dest.write_bytes(get(url, tries=3))
        r["sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
        if n % 300 == 0:
            print(f"photos {n}/{len(rows)}", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(rows)} photos of {len({r['scientific'] for r in rows})} species in "
          f"{len({r['country_name'] for r in rows})} countries ({sum(not r['country'] for r in rows)} photos with no GBIF country code) -> {OUT}")


if __name__ == "__main__":
    main()
