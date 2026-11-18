"""More photos for the regions the owner shoots in: research-grade iNaturalist bird photos, up to N per species.

  ./run distill/fetch_inat.py --out data/inat-raw --plan     # the species lists and an estimate (a few API calls)
  ./run distill/fetch_inat.py --out data/inat-raw            # list photos (1-2 API calls per species), then download
  ./run distill/fetch_inat.py --check                        # the golden-set skip, no network

Species are picked by region, not by GPS: iNaturalist's own bird list of each country in --places.
A species' photos come from those countries first (random order, so every season and year), and from
the rest of the world only when the countries have fewer than N. One photo per observation.

OUT/species.csv   one row per species: which regions list it, and how many observations each has
OUT/manifest.csv  one row per photo: licence, attribution, date, iNaturalist place ids (country, state...)
OUT/SPECIES/OBSERVATION_PHOTO.jpg

Rerun to resume: listed species and downloaded files are skipped. Photos and observations of
gardener's golden set are never listed. Downloads hold --gb-per-hour and stop at --max-gb per run
(iNaturalist asks for under 5 GB an hour and 24 GB a day of media, and about one API call a second).
"""
from __future__ import annotations

import argparse
import csv
import http.client
import json
import math
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://api.inaturalist.org/v1/"
UA = "bird-photo-fetch/0.1 (personal research)"
LICENSES = ("cc0", "cc-by", "cc-by-nc")
BIRDS = 3                                            # iNaturalist taxon id of Aves
PAGE = 200                                           # the API's largest page of observations
FIELDS = ["path", "photo_id", "observation_id", "taxon_id", "scientific", "license", "attribution", "observed_on", "place_ids", "url"]
GOLDEN = Path(__file__).resolve().parents[2] / "bioscan" / "data" / "inat" / "groundtruth-inat.csv"


def golden_ids() -> set[str]:
    """Observation and photo ids of the golden set (its files are OBSERVATION_PHOTO.jpg)."""
    with open(GOLDEN, newline="") as f:
        return {part for r in csv.DictReader(f) for part in Path(r["path"]).stem.split("_")}


def get(url: str, tries: int = 1) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or k == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if k == tries - 1:
                raise
        time.sleep(60 * (k + 1))
    raise AssertionError("unreachable")


class Api:
    """Calls start at least a second apart, whatever the number of threads; retries on 429 and 5xx."""

    def __init__(self) -> None:
        self.last, self.lock = 0.0, threading.Lock()

    def __call__(self, path: str, **q) -> dict:
        with self.lock:
            time.sleep(max(0.0, 1.0 - (time.monotonic() - self.last)))
            self.last = time.monotonic()
        return json.loads(get(API + path + "?" + urllib.parse.urlencode(q), tries=5))


def species(api: Api, places: dict[str, int], out: Path) -> list[dict]:
    """Every bird species iNaturalist lists for the places, with its observation count in each. Cached in species.csv."""
    path = out / "species.csv"
    if not path.exists():
        found: dict[int, dict] = {}
        for code, place in places.items():
            page = 1
            while True:
                results = api("observations/species_counts", taxon_id=BIRDS, place_id=place, quality_grade="research",
                              photo_license=",".join(LICENSES), per_page=500, page=page)["results"]
                for r in results:
                    t = r["taxon"]
                    if t.get("rank") == "species":
                        row = found.setdefault(t["id"], {"taxon_id": t["id"], "scientific": t["name"], "common": t.get("preferred_common_name", ""), "regions": {}})
                        row["regions"][code] = r["count"]
                if len(results) < 500:
                    break
                page += 1
            print(f"{code}: {sum(code in r['regions'] for r in found.values())} species", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, ["taxon_id", "scientific", "common", "regions"])
            w.writeheader()
            w.writerows({**r, "regions": ";".join(f"{c}:{n}" for c, n in r["regions"].items())} for r in sorted(found.values(), key=lambda r: r["scientific"]))
    with open(path, newline="") as f:
        return [{**r, "regions": {c: int(n) for c, n in (x.split(":") for x in r["regions"].split(";"))}} for r in csv.DictReader(f)]


def pick(results: list[dict], sp: dict, skip: set[str], seen: set[int], room: int) -> list[dict]:
    """One licensed photo per new observation, never one of the golden set's."""
    rows = []
    for obs in results:
        photo = next((p for p in obs.get("photos") or [] if (p.get("license_code") or "") in LICENSES and p.get("url")), None)
        if len(rows) >= room or not photo or obs["id"] in seen or str(obs["id"]) in skip or str(photo["id"]) in skip:
            continue
        seen.add(obs["id"])
        ext = photo["url"].rsplit(".", 1)[-1].split("?")[0].lower()
        rows.append({"path": f"{sp['scientific'].replace(' ', '_')}/{obs['id']}_{photo['id']}.{ext}", "photo_id": photo["id"],
                     "observation_id": obs["id"], "taxon_id": sp["taxon_id"], "scientific": sp["scientific"],
                     "license": photo["license_code"], "attribution": photo.get("attribution", ""), "observed_on": obs.get("observed_on") or "",
                     "place_ids": " ".join(map(str, obs.get("place_ids") or [])), "url": photo["url"].replace("/square.", "/medium.")})
    return rows


def listing(api: Api, sp: dict, places: dict[str, int], cap: int, skip: set[str]) -> list[dict]:
    q = {"taxon_id": sp["taxon_id"], "quality_grade": "research", "photo_license": ",".join(LICENSES), "photos": "true",
         "per_page": PAGE, "order_by": "random"}
    here = ",".join(str(places[c]) for c in sp["regions"])
    seen: set[int] = set()
    rows: list[dict] = []
    for _ in range(math.ceil(cap / PAGE)):           # the regions first
        results = api("observations", **q, place_id=here)["results"]
        rows += pick(results, sp, skip, seen, cap - len(rows))
        if len(rows) >= cap or len(results) < PAGE:
            break
    if len(rows) < cap and sum(sp["regions"].values()) < cap:      # the regions ran dry: the rest of the world
        rows += pick(api("observations", **q)["results"], sp, skip, seen, cap - len(rows))
    return rows


class Budget:
    """Holds the download to a number of GB an hour and stops it at a total; gives up when the host refuses everything."""

    def __init__(self, gb_per_hour: float, max_gb: float) -> None:
        self.rate, self.max, self.t0 = gb_per_hour * 1e9 / 3600, max_gb * 1e9, time.monotonic()
        self.bytes = self.files = self.fails = 0
        self.lock = threading.Lock()

    def spent(self, n: int) -> None:
        with self.lock:
            self.bytes, self.files, self.fails = self.bytes + n, self.files + 1, 0
            lag = self.bytes / self.rate - (time.monotonic() - self.t0)
            if self.files % 2000 == 0:
                print(f"{self.files} photos, {self.bytes / 1e9:.2f} GB, {(time.monotonic() - self.t0) / 3600:.2f} h", flush=True)
        if lag > 0:
            time.sleep(lag)

    def failed(self) -> None:
        with self.lock:
            self.fails += 1

    def stop(self) -> bool:
        return self.bytes >= self.max or self.fails >= 200


def download(out: Path, rows: list[dict], budget: Budget) -> None:
    def fetch(r: dict) -> None:
        p = out / r["path"]
        if budget.stop() or p.exists():
            return
        try:
            data = get(r["url"])
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".part")
            tmp.write_bytes(data)
            tmp.rename(p)                             # a killed run leaves no half-written photo
            budget.spent(len(data))
        except (OSError, http.client.HTTPException) as e:     # a photo gone from the bucket: skip it, the manifest still lists it
            budget.failed()
            print(f"skip {r['url']}: {e}", flush=True)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(fetch, rows))


def check() -> None:
    """The golden-set skip, on made-up observations."""
    sp = {"scientific": "Turdus migratorius", "taxon_id": 12727}
    obs = lambda o, p, lic="cc-by": {"id": o, "photos": [{"id": p, "license_code": lic, "url": f"https://x/photos/{p}/square.jpeg"}]}
    got = pick([obs(1, 10), obs(2, 20), obs(3, 30), obs(4, 40, None), obs(5, 50), obs(5, 51), obs(6, 60)], sp, {"1", "30"}, set(), 2)
    assert [r["observation_id"] for r in got] == [2, 5], got          # 1: golden observation, 3: golden photo, 4: no licence, 6: over the cap
    assert got[0]["url"].endswith("/photos/20/medium.jpeg") and got[0]["path"] == "Turdus_migratorius/2_20.jpeg", got
    assert len(golden_ids()) > 2000
    print("check ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--places", default="US=1,CA=6712,CN=6903", help="CODE=iNaturalist place id, comma separated")
    ap.add_argument("--per-species", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="only the first N species (a trial run)")
    ap.add_argument("--gb-per-hour", type=float, default=4.0)
    ap.add_argument("--max-gb", type=float, default=20.0)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    out, api = Path(a.out), Api()
    places = {c: int(p) for c, p in (x.split("=") for x in a.places.split(","))}
    sps = species(api, places, out)
    est = sum(min(sum(s["regions"].values()), a.per_species) for s in sps)
    print(f"{len(sps)} species: " + ", ".join(f"{c} {sum(c in s['regions'] for s in sps)}" for c in places)
          + f"; at least {est} photos at {a.per_species} a species ({sum(sum(s['regions'].values()) < a.per_species for s in sps)} species have fewer in these regions)", flush=True)
    if a.plan:
        return
    sps = sps[:a.limit] if a.limit else sps
    skip, done_path, manifest = golden_ids(), out / "listed.txt", out / "manifest.csv"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    new = not manifest.exists()
    with open(manifest, "a", newline="") as f, open(done_path, "a") as d:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        lock = threading.Lock()

        def one(sp: dict) -> None:
            try:
                rows = listing(api, sp, places, a.per_species, skip)
            except (OSError, http.client.HTTPException, ValueError, KeyError) as e:    # left unlisted: the next run tries it again
                print(f"{sp['scientific']}: {e}", flush=True)
                return
            with lock:
                w.writerows(rows)
                f.flush()
                d.write(sp["taxon_id"] + "\n")
                d.flush()
                done.add(sp["taxon_id"])
                if len(done) % 50 == 0:
                    print(f"listed {len(done)}/{len(sps)}  {sp['scientific']}: {len(rows)}", flush=True)
        with ThreadPoolExecutor(4) as pool:           # a call takes seconds to answer; the Api still starts one a second
            list(pool.map(one, [sp for sp in sps if sp["taxon_id"] not in done]))
    with open(manifest, newline="") as f:
        rows = list({r["path"]: r for r in csv.DictReader(f)}.values())
    assert not any(r["observation_id"] in skip or r["photo_id"] in skip for r in rows), "a golden photo is in the manifest"
    random.Random(7).shuffle(rows)                    # an interrupted download is still an even sample of every species
    print(f"{len(rows)} photos listed, 0 from the golden set; downloading at most {a.max_gb} GB at {a.gb_per_hour} GB/h", flush=True)
    budget = Budget(a.gb_per_hour, a.max_gb)
    download(out, rows, budget)
    have = sum((out / r["path"]).exists() for r in rows)
    why = "the host refused 200 photos in a row" if budget.fails >= 200 else "this run's size limit reached, rerun to continue" if have < len(rows) and budget.bytes >= budget.max else "done"
    print(f"{why}: {have}/{len(rows)} photos on disk, {budget.bytes / 1e9:.2f} GB this run", flush=True)


if __name__ == "__main__":
    main()
