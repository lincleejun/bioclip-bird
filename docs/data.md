# Data

What each dataset is, where it came from, what it is used for here, and under which terms. Counts are computed
from the files named; a number from the project's working notes is marked *notes*.

## Evaluation set (`eval/groundtruth.csv`)

gardener's golden set, used for every pipeline number in [results.md](results.md). Never trained on.

**How it was chosen.** `gardener-engine gt inat --place california --taxa data/taxa.csv --per-species 25`
(gardener's `gardener/cli/gt.py`, September 2026): for each of 65 species in gardener's `data/taxa.csv` (common
California birds and mammals, plus caribou, moose and grizzly bear), the first 25 research-grade iNaturalist
observations the API returned in California with a CC0, CC BY or CC BY-NC photo, falling back to anywhere when
California had none; one photo per observation, "medium" size. On 2026-11-13 gardener removed 54 rows
(gardener `docs/results.md`, issue #156): all 25 *Alces alces* rows, which were Elk Kelp, and 29 photos showing
only tracks, scat, a bone or feathers. 1,571 photos remain.

| | Photos | Species | Photos per species | Photographers | CC BY-NC / CC BY / CC0 |
|---|---|---|---|---|---|
| Birds | 1,046 | 42 | 25 for 39 species, 24 for 2, 23 for 1 | 470 | 925 / 85 / 36 |
| Mammals | 525 | 22 | 25 for 14 species, 18-24 for 8 | 363 | 477 / 35 / 13 |
| All | 1,571 | 64 | | | 1,402 / 120 / 49 |

Photographers = distinct names in the attribution column.

- **Image size:** iNaturalist "medium": the long side is 500 px for 1,550 photos (the smallest file's long side
  is 337 px); short side median 375 px.
- **Where:** all 1,046 bird photos and 479 mammal photos fall inside a rough California box (lat 32.4-42.1,
  lon -124.6 to -114.1); of the other mammals, 5 are in a rough Alaska box, 18 elsewhere in the Americas, 22
  outside the Americas, 1 has no coordinates.
- **When:** 1,494 photos were taken in 2026 (1,337 in September 2026), 35 in 2025, the remaining 42 between 2003
  and 2024.
- **Labels:** the observation's research-grade taxon. Scoring renames a label to the AviList/MDD name through
  gardener's `data/names/synonyms.csv` (25 bird photos gain a hit that way in the original run).
- **Kept out of training:** every photo and observation id of this set is excluded from all training photos,
  and so is any photo by the same photographer of the same species on the same day (likely the same bird under
  another observation; `distill/embed.py`). Recorded crops of this set (`data/rec/golden`) are used only to
  measure.
- **Known biases:** one source (iNaturalist, mostly phone and small-camera photos at 500 px), one region
  (California) and one season (September 2026) for the birds; 42 common bird species out of 11,172 in the names
  matrix, so it says little about rare or look-alike species elsewhere. The student models' extra training photos
  also come from iNaturalist, which favours them here.
- **Photos are not redistributed.** `eval/groundtruth.csv` keeps each photo's source URL, licence and attribution
  and a sha256; `python eval/fetch_photos.py` downloads them from iNaturalist (one request a second; checked on 3 photos
  against the live API).

## Worldwide evaluation set (`eval/inat-global.csv`)

The set to read first for "how well does it name a bird anywhere". Never trained on.
`./run eval/build_global.py` (one API call or photo a second; API answers cached in `data/`).

**How it was chosen.** 1,200 species drawn at random (seed 2026) from all 11,037 active, non-extinct bird species
iNaturalist lists. For each, research-grade observations made anywhere in 2026 with a CC0 / CC BY / CC BY-NC photo;
of the newest 60, up to 3 picked at random, each by a different observer; one photo per observation at "medium" size
(500 px). Left out as for the second set: every observation and photo the students trained on, the first set's, and
any observation by the same person of the same species on the same day as a training photo. 962 of the 1,200 species
had a usable observation. Each photo's country is its observation's iNaturalist country (38 photos have none, mostly
at sea).

| Region (rough coordinate boxes) | Photos | Species |
|---|---|---|
| Americas | 1,227 | 441 |
| Asia | 621 | 254 |
| Africa | 493 | 193 |
| Oceania | 195 | 75 |
| Europe | 96 | 44 |
| All | 2,632 | 962 (3 photos for 808, 2 for 54, 1 for 100) |

- **Countries:** 138; most photos from Brazil 242, the US 149, Australia 148, South Africa 115, Colombia 114,
  India 114, Indonesia 112. Observers: 1,807.
- **When:** every month of 2026 so far, 152-442 photos a month (most in October and November).
- **Licences:** CC BY-NC 2,262, CC BY 282, CC0 88.
- **Known biases:** a species needed a 2026 observation, so the set leans to commoner species (238 of 1,200 drawn
  had none); Europe has few species, as it has few bird species; still one source (iNaturalist, 500 px).
- **Photos are not redistributed:** `python eval/fetch_photos.py inat-global.csv` downloads them.

## Second evaluation set (`eval/inat2026.csv`)

Built to test what the first set cannot: more species, three countries, uncommon birds. Never trained on.
`./run eval/build_eval2.py` (one API call or photo a second; the API answers are cached in `data/`).

**How it was chosen.** 500 species drawn at random (seed 2026) from iNaturalist's bird lists of the US, Canada and
China (`data/inat-raw/species.csv`, 2,179 species); each species was given one of the countries whose list has it,
at random. For each, the newest research-grade observations of 2026 in that country (iNaturalist API, newest first,
60 per species), up to 3, each by a different observer, with a CC0 / CC BY / CC BY-NC photo; one photo per
observation at "medium" size (500 px). Left out: every observation and photo the students trained on, the first
set's, and any observation by the same person of the same species on the same day as a training photo. 2026 photos
are newer than BioCLIP 2.5's training data. 228 of the 500 species had at least one usable observation.

| | Photos | Species | Photos per species | Observers |
|---|---|---|---|---|
| US | 329 | 114 | | |
| Canada | 157 | 53 | | |
| China | 151 | 61 | | |
| All | 637 | 228 | 3 for 201 species, 2 for 7, 1 for 20 | 532 |

- **Licences:** CC BY-NC 588, CC BY 40, CC0 9.
- **When:** all 2026; newest first, so 393 photos from November and 100 from October, 2-61 for each earlier month.
- **Overlap with the first set:** 11 species.
- **Known biases:** a species needed 2026 observations in its country, so the set leans to commoner species (half
  of the 500 drawn had none); taking the newest observations puts most photos in late autumn; still one source
  (iNaturalist, 500 px).
- **Photos are not redistributed:** `python eval/fetch_photos.py inat2026.csv` downloads them and checks each
  against its sha256 (checked on 3 photos).

## CUB-200-2011

Caltech-UCSD Birds-200-2011: 11,788 photos of 200 bird species, each with one bird box and 15 part
points, and an official train/test split (5,994 / 5,794). Used here for the box comparison (test split, see
results.md B) and to train and test the beak/eye head (bioclip-bird-heads). Terms: non-commercial research and
educational use only. Obtain it from the Caltech data site (CaltechDATA, "Caltech-UCSD Birds-200-2011") and unpack
it as `data/cub/CUB_200_2011`; it is not redistributed here.

## Student training photos (not released)

- **Local reference photos**: 116,247 photos of 9,445 species (median 12 a species; `data/birding`), licensed for
  personal research only (*notes*). Not redistributed; the students trained on them are not released either.
- **iNaturalist by region** (`distill/fetch_inat.py`): every bird species on iNaturalist's lists for the US, Canada
  and China (2,179 species), up to 200 research-grade photos each (those countries first, the rest of the world when
  short), one photo per observation, random order, 500 px, never a golden-set photo or observation. 381,148
  listed (CC BY-NC 333,171 / CC BY 39,362 / CC0 8,615); the first day's download stopped at its 20 GB cap with
  186,524 on disk (*notes*); 186,504 got teacher vectors. `distill/quality.py` then dropped 3,376 (1.8 %): 3,273
  whose label the teacher could not see at all in its top 50 (a sample of 48 was mostly feathers, nests,
  spectrograms, far specks and empty scenes; *notes*) and 104 duplicates. 183,128 used, 2,175 species
  (CC BY-NC 159,997 / CC BY 19,044 / CC0 4,087).

## Flight photos

4,876 iNaturalist photos labelled by their observers as flying or not; see bioclip-bird-heads `docs/heads.md`.

## Bird names matrix

11,172 species of AviList v2026 (CC BY 4.0). 9,416 rows are the official BioCLIP 2.5 text vectors of
TreeOfLife-200M (CC0; 9,373 by exact name, 43 through a synonym), 1,756 were encoded with BioCLIP 2.5's text tower.
Details and citation: [MODELS.md](../MODELS.md).

## The owner's own photos

A few numbers in the docs come from the owner's own shoots (marked "private shoot"): 49 raw frames for the
small-bird detector check, 94 detector crops for the heads. They were only looked at, never trained on, and are not
published.
