# gardener-engine eval report

- groundtruth: data/inat/groundtruth-inat.csv
- preds: preds.ndjson
- images: 1571
- geo: True
- preds schema: 1
- complete: True
- synonyms: data/names/synonyms.csv (gardener)
- generated: 2026-11-17 10:52:57
- wall_s: 468.3
- engine: {"detail_edge": 3072, "models": {"detect": "google/owlv2-base-patch16-ensemble@cfd3195ba4ea", "gate": "google/siglip2-base-patch16-224@75de2d55ec2d", "geo": "birdnet-geo-3.0", "label_maps": {"bird": "avilist_map.csv@55d1aa841f8d", "mammal": "mdd_map.csv@758a82fafbbd"}, "names": {"bird": "avilist-2026@a4f7858b14c15128", "mammal": "mdd-2025@3eab033f008ac62f", "other_animal": "tol200m-animalia@4674544e7661da91"}, "priors": {"bird": "birdnet-geo-3.0", "mammal": "birdnet-geo-3.0", "other_animal": null}, "species": "imageomics/bioclip-2.5-vith14@6e3d04e3d652", "taxonomy": {"bird": "AviList v2026", "mammal": "MDD v2.5", "other_animal": "TreeOfLife-200M @ 5f2dc493"}}, "settings": "79a2ce363c57", "version": "0.1.0"}

## Metrics by tier × kind

| tier | kind | n | failed | gate acc | detect | Top-1 | Top-5 | coverage | precision | decode ms (mean/median) | identify ms (mean/median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inat | bird | 1046 | 0 | 97.1% | 99.0% | 92.4% | 96.9% | 93.6% | 97.0% | 2 / 2 | 258 / 245 |
| inat | mammal | 525 | 0 | 94.9% | 98.5% | 91.4% | 95.4% | 88.0% | 96.5% | 2 / 2 | 346 / 294 |

Definitions: Top-1/Top-5 use the highest-`score` box; coverage = best box at `level == species`; precision = Top-1 hit rate among those; failed images count as misses everywhere.

Top-1 hits gained by synonym normalisation of the truth (miss -> hit): inat/bird 25, inat/mammal 0

No box, by whole-frame gate class (none/person: the gate missed the animal; else the detector did): inat/bird bird 3, mammal 3, person 1; inat/mammal none 2, mammal 2, bird 1

## Confusion Top-10 — inat / bird

| truth | predicted | count |
|---|---|---|
| Tyto furcata | Tyto alba | 18 |
| Athene cunicularia | (no box) | 4 |
| Nycticorax nycticorax | Nyctanassa violacea | 3 |
| Astur cooperii | Accipiter striatus | 2 |
| Haliaeetus leucocephalus | Nucifraga columbiana | 2 |
| Megascops kennicottii | Otus scops | 2 |
| Fulica americana | Fulica atra | 2 |
| Junco hyemalis | (no box) | 2 |
| Buteo jamaicensis | Aviceda subcristata | 1 |
| Buteo lineatus | Nycticorax nycticorax | 1 |

## Confusion Top-10 — inat / mammal

| truth | predicted | count |
|---|---|---|
| Cervus canadensis | Cervus elaphus | 2 |
| Cervus canadensis | Cervus albirostris | 2 |
| Zalophus californianus | Zalophus japonicus | 2 |
| Phoca vitulina | Phoca largha | 2 |
| Rangifer tarandus | Pseudois nayaur | 2 |
| Ursus arctos | Ursus maritimus | 2 |
| Odocoileus hemionus | Odocoileus pandora | 1 |
| Canis latrans | Rupicapra rupicapra | 1 |
| Procyon lotor | Mephitis mephitis | 1 |
| Procyon lotor | Felis catus | 1 |
