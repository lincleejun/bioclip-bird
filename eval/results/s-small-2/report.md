# gardener-engine eval report

- groundtruth: data/inat/groundtruth-inat.csv
- preds: preds.ndjson
- images: 1571
- geo: True
- preds schema: 1
- complete: True
- synonyms: data/names/synonyms.csv (gardener)
- generated: 2026-11-17 13:15:23
- wall_s: 257.6
- engine: {"detail_edge": 3072, "models": {"detect": "google/owlv2-base-patch16-ensemble@cfd3195ba4ea", "gate": "google/siglip2-base-patch16-224@75de2d55ec2d", "geo": "birdnet-geo-3.0", "label_maps": {"bird": "avilist_map.csv@55d1aa841f8d", "mammal": "mdd_map.csv@758a82fafbbd"}, "names": {"bird": "avilist-2026@a4f7858b14c15128", "mammal": "mdd-2025@3eab033f008ac62f", "other_animal": "tol200m-animalia@4674544e7661da91"}, "priors": {"bird": "birdnet-geo-3.0", "mammal": "birdnet-geo-3.0", "other_animal": null}, "species": "imageomics/bioclip-2.5-vith14@6e3d04e3d652", "taxonomy": {"bird": "AviList v2026", "mammal": "MDD v2.5", "other_animal": "TreeOfLife-200M @ 5f2dc493"}}, "settings": "79a2ce363c57", "version": "0.1.0"}

## Metrics by tier × kind

| tier | kind | n | failed | gate acc | detect | Top-1 | Top-5 | coverage | precision | decode ms (mean/median) | identify ms (mean/median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inat | bird | 1046 | 0 | 97.1% | 99.2% | 60.0% | 78.7% | 67.0% | 76.2% | 1 / 1 | 150 / 146 |
| inat | mammal | 525 | 0 | 94.9% | 0.6% | 0.0% | 0.0% | 23.4% | 0.0% | 2 / 1 | 169 / 162 |

Definitions: Top-1/Top-5 use the highest-`score` box; coverage = best box at `level == species`; precision = Top-1 hit rate among those; failed images count as misses everywhere.

Top-1 hits gained by synonym normalisation of the truth (miss -> hit): inat/bird 18, inat/mammal 0

No box, by whole-frame gate class (none/person: the gate missed the animal; else the detector did): inat/bird bird 3, mammal 3, person 1; inat/mammal none 2, mammal 2, bird 1

## Confusion Top-10 — inat / bird

| truth | predicted | count |
|---|---|---|
| Branta canadensis | Branta hutchinsii | 17 |
| Tyto furcata | Tyto alba | 14 |
| Astur cooperii | Accipiter striatus | 10 |
| Ardea alba | Egretta thula | 8 |
| Buteo jamaicensis | Buteo swainsoni | 6 |
| Corvus corax | Corvus brachyrhynchos | 6 |
| Nannopterum auritum | Nannopterum brasilianum | 6 |
| Aphelocoma californica | Aphelocoma woodhouseii | 5 |
| Calypte anna | Archilochus alexandri | 5 |
| Athene cunicularia | (no box) | 4 |

## Confusion Top-10 — inat / mammal

| truth | predicted | count |
|---|---|---|
| Marmota flaviventris | Tinamotis pentlandii | 7 |
| Marmota flaviventris | Attagis gayi | 6 |
| Lynx rufus | Bubo virginianus | 5 |
| Urocyon cinereoargenteus | Bubo virginianus | 5 |
| Odocoileus hemionus | Dromaius novaehollandiae | 4 |
| Phoca vitulina | Calidris virgata | 4 |
| Mirounga angustirostris | Larus heermanni | 4 |
| Puma concolor | Tyto alba | 4 |
| Rangifer tarandus | Struthio camelus | 4 |
| Odocoileus hemionus | Meleagris gallopavo | 3 |
