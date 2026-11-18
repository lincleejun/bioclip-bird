# gardener-engine eval report

- groundtruth: data/inat/groundtruth-inat.csv
- preds: preds.ndjson.gz
- images: 1571
- geo: False
- preds schema: 1
- complete: True
- synonyms: data/names/synonyms.csv (gardener)
- generated: 2026-11-17 22:09:41
- wall_s: 941.5
- engine: {"detail_edge": 3072, "models": {"detect": "google/owlv2-base-patch16-ensemble@cfd3195ba4ea", "gate": "google/siglip2-base-patch16-224@75de2d55ec2d", "geo": "birdnet-geo-3.0", "label_maps": {"bird": "avilist_map.csv@55d1aa841f8d", "mammal": "mdd_map.csv@758a82fafbbd"}, "names": {"bird": "avilist-2026@a4f7858b14c15128", "mammal": "mdd-2025@3eab033f008ac62f", "other_animal": "tol200m-animalia@4674544e7661da91"}, "priors": {"bird": "birdnet-geo-3.0", "mammal": "birdnet-geo-3.0", "other_animal": null}, "species": "imageomics/bioclip-2.5-vith14@6e3d04e3d652", "taxonomy": {"bird": "AviList v2026", "mammal": "MDD v2.5", "other_animal": "TreeOfLife-200M @ 5f2dc493"}}, "settings": "79a2ce363c57", "version": "0.1.0"}

## Metrics by tier × kind

| tier | kind | n | failed | gate acc | detect | Top-1 | Top-5 | coverage | precision | decode ms (mean/median) | identify ms (mean/median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inat | bird | 1046 | 0 | 97.1% | 99.0% | 87.0% | 96.1% | 93.3% | 91.9% | 2 / 1 | 569 / 532 |
| inat | mammal | 525 | 0 | 94.9% | 98.5% | 85.7% | 94.3% | 94.7% | 89.7% | 2 / 1 | 631 / 400 |

Definitions: Top-1/Top-5 use the highest-`score` box; coverage = best box at `level == species`; precision = Top-1 hit rate among those; failed images count as misses everywhere.

Top-1 hits gained by synonym normalisation of the truth (miss -> hit): inat/bird 24, inat/mammal 0

No box, by whole-frame gate class (none/person: the gate missed the animal; else the detector did): inat/bird bird 3, mammal 3, person 1; inat/mammal none 2, mammal 2, bird 1

## Confusion Top-10 — inat / bird

| truth | predicted | count |
|---|---|---|
| Circus hudsonius | Circus cyaneus | 22 |
| Tyto furcata | Tyto alba | 18 |
| Athene cunicularia | (no box) | 4 |
| Elanus leucurus | Elanus axillaris | 3 |
| Nycticorax nycticorax | Nyctanassa violacea | 3 |
| Pelecanus occidentalis | Pelecanus thagus | 3 |
| Nannopterum auritum | Urile penicillatus | 3 |
| Astur cooperii | Accipiter striatus | 2 |
| Circus hudsonius | Circus macrourus | 2 |
| Megascops kennicottii | Otus scops | 2 |

## Confusion Top-10 — inat / mammal

| truth | predicted | count |
|---|---|---|
| Cervus canadensis | Cervus elaphus | 23 |
| Ursus americanus | Ursus arctos | 2 |
| Mirounga angustirostris | Mirounga leonina | 2 |
| Rangifer tarandus | Pseudois nayaur | 2 |
| Ursus arctos | Ursus maritimus | 2 |
| Odocoileus hemionus | Mazama nanus | 1 |
| Canis latrans | Rupicapra rupicapra | 1 |
| Canis latrans | Puma concolor | 1 |
| Procyon lotor | Meles meles | 1 |
| Procyon lotor | Felis chaus | 1 |
