# gardener-engine eval report

- groundtruth: data/inat/groundtruth-inat.csv
- preds: preds.ndjson
- images: 1571
- geo: True
- preds schema: 1
- complete: True
- synonyms: data/names/synonyms.csv (gardener)
- generated: 2026-11-17 17:08:36
- wall_s: 166.9
- engine: {"detail_edge": 3072, "models": {"detect": "google/owlv2-base-patch16-ensemble@cfd3195ba4ea", "gate": "google/siglip2-base-patch16-224@75de2d55ec2d", "geo": "birdnet-geo-3.0", "label_maps": {"bird": "avilist_map.csv@55d1aa841f8d", "mammal": "mdd_map.csv@758a82fafbbd"}, "names": {"bird": "avilist-2026@a4f7858b14c15128", "mammal": "mdd-2025@3eab033f008ac62f", "other_animal": "tol200m-animalia@4674544e7661da91"}, "priors": {"bird": "birdnet-geo-3.0", "mammal": "birdnet-geo-3.0", "other_animal": null}, "species": "imageomics/bioclip-2.5-vith14@6e3d04e3d652", "taxonomy": {"bird": "AviList v2026", "mammal": "MDD v2.5", "other_animal": "TreeOfLife-200M @ 5f2dc493"}}, "settings": "79a2ce363c57", "version": "0.1.0"}

## Metrics by tier × kind

| tier | kind | n | failed | gate acc | detect | Top-1 | Top-5 | coverage | precision | decode ms (mean/median) | identify ms (mean/median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inat | bird | 1046 | 0 | 97.1% | 91.2% | 86.2% | 89.6% | 87.3% | 97.4% | 1 / 1 | 118 / 112 |
| inat | mammal | 525 | 0 | 94.9% | 11.8% | 10.7% | 11.2% | 11.2% | 93.2% | 1 / 1 | 67 / 68 |

Definitions: Top-1/Top-5 use the highest-`score` box; coverage = best box at `level == species`; precision = Top-1 hit rate among those; failed images count as misses everywhere.

Top-1 hits gained by synonym normalisation of the truth (miss -> hit): inat/bird 24, inat/mammal 0

No box, by whole-frame gate class (none/person: the gate missed the animal; else the detector did): inat/bird bird 75, mammal 10, other_animal 4, person 1; inat/mammal mammal 443, bird 9, other_animal 7, none 4

## Confusion Top-10 — inat / bird

| truth | predicted | count |
|---|---|---|
| Bubo virginianus | (no box) | 11 |
| Tyto furcata | (no box) | 11 |
| Tyto furcata | Tyto alba | 11 |
| Megascops kennicottii | (no box) | 10 |
| Athene cunicularia | (no box) | 9 |
| Cyanocitta stelleri | (no box) | 5 |
| Dryobates nuttallii | (no box) | 5 |
| Elanus leucurus | (no box) | 4 |
| Melanerpes formicivorus | (no box) | 4 |
| Calypte anna | (no box) | 4 |

## Confusion Top-10 — inat / mammal

| truth | predicted | count |
|---|---|---|
| Otospermophilus beecheyi | (no box) | 25 |
| Cervus canadensis | (no box) | 25 |
| Canis latrans | (no box) | 24 |
| Sciurus niger | (no box) | 24 |
| Didelphis virginiana | (no box) | 24 |
| Marmota flaviventris | (no box) | 24 |
| Odocoileus hemionus | (no box) | 23 |
| Urocyon cinereoargenteus | (no box) | 23 |
| Rangifer tarandus | (no box) | 23 |
| Sciurus griseus | (no box) | 22 |
