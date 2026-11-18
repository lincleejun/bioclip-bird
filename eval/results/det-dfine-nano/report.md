# gardener-engine eval report

- groundtruth: data/inat/groundtruth-inat.csv
- preds: preds.ndjson
- images: 1571
- geo: True
- preds schema: 1
- complete: True
- synonyms: data/names/synonyms.csv (gardener)
- generated: 2026-11-17 16:52:20
- wall_s: 259.3
- engine: {"detail_edge": 3072, "models": {"detect": "google/owlv2-base-patch16-ensemble@cfd3195ba4ea", "gate": "google/siglip2-base-patch16-224@75de2d55ec2d", "geo": "birdnet-geo-3.0", "label_maps": {"bird": "avilist_map.csv@55d1aa841f8d", "mammal": "mdd_map.csv@758a82fafbbd"}, "names": {"bird": "avilist-2026@a4f7858b14c15128", "mammal": "mdd-2025@3eab033f008ac62f", "other_animal": "tol200m-animalia@4674544e7661da91"}, "priors": {"bird": "birdnet-geo-3.0", "mammal": "birdnet-geo-3.0", "other_animal": null}, "species": "imageomics/bioclip-2.5-vith14@6e3d04e3d652", "taxonomy": {"bird": "AviList v2026", "mammal": "MDD v2.5", "other_animal": "TreeOfLife-200M @ 5f2dc493"}}, "settings": "79a2ce363c57", "version": "0.1.0"}

## Metrics by tier × kind

| tier | kind | n | failed | gate acc | detect | Top-1 | Top-5 | coverage | precision | decode ms (mean/median) | identify ms (mean/median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inat | bird | 1046 | 0 | 97.1% | 97.9% | 91.1% | 95.0% | 91.9% | 97.1% | 1 / 1 | 140 / 124 |
| inat | mammal | 525 | 0 | 94.9% | 40.0% | 36.6% | 39.0% | 37.5% | 93.9% | 2 / 2 | 190 / 170 |

Definitions: Top-1/Top-5 use the highest-`score` box; coverage = best box at `level == species`; precision = Top-1 hit rate among those; failed images count as misses everywhere.

Top-1 hits gained by synonym normalisation of the truth (miss -> hit): inat/bird 24, inat/mammal 0

No box, by whole-frame gate class (none/person: the gate missed the animal; else the detector did): inat/bird bird 13, mammal 5, person 1; inat/mammal mammal 305, none 4, other_animal 4

## Confusion Top-10 — inat / bird

| truth | predicted | count |
|---|---|---|
| Tyto furcata | Tyto alba | 17 |
| Megascops kennicottii | (no box) | 4 |
| Nycticorax nycticorax | Nyctanassa violacea | 3 |
| Bubo virginianus | (no box) | 2 |
| Tyto furcata | (no box) | 2 |
| Tyto furcata | Nycticorax nycticorax | 2 |
| Athene cunicularia | (no box) | 2 |
| Athene cunicularia | Callipepla gambelii | 2 |
| Pelecanus occidentalis | Pelecanus erythrorhynchos | 2 |
| Fulica americana | (no box) | 2 |

## Confusion Top-10 — inat / mammal

| truth | predicted | count |
|---|---|---|
| Canis latrans | (no box) | 23 |
| Odocoileus hemionus | (no box) | 21 |
| Cervus canadensis | (no box) | 20 |
| Ursus arctos | (no box) | 20 |
| Marmota flaviventris | (no box) | 19 |
| Didelphis virginiana | (no box) | 18 |
| Puma concolor | (no box) | 18 |
| Rangifer tarandus | (no box) | 17 |
| Lynx rufus | (no box) | 16 |
| Ursus americanus | (no box) | 16 |
