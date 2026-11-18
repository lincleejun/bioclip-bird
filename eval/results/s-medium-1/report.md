# gardener-engine eval report

- groundtruth: data/inat/groundtruth-inat.csv
- preds: preds.ndjson
- images: 1571
- geo: True
- preds schema: 1
- complete: True
- synonyms: data/names/synonyms.csv (gardener)
- generated: 2026-11-17 03:43:47
- wall_s: 265.2
- engine: {"detail_edge": 3072, "models": {"detect": "google/owlv2-base-patch16-ensemble@cfd3195ba4ea", "gate": "google/siglip2-base-patch16-224@75de2d55ec2d", "geo": "birdnet-geo-3.0", "label_maps": {"bird": "avilist_map.csv@55d1aa841f8d", "mammal": "mdd_map.csv@758a82fafbbd"}, "names": {"bird": "avilist-2026@a4f7858b14c15128", "mammal": "mdd-2025@3eab033f008ac62f", "other_animal": "tol200m-animalia@4674544e7661da91"}, "priors": {"bird": "birdnet-geo-3.0", "mammal": "birdnet-geo-3.0", "other_animal": null}, "species": "imageomics/bioclip-2.5-vith14@6e3d04e3d652", "taxonomy": {"bird": "AviList v2026", "mammal": "MDD v2.5", "other_animal": "TreeOfLife-200M @ 5f2dc493"}}, "settings": "79a2ce363c57", "version": "0.1.0"}

## Metrics by tier × kind

| tier | kind | n | failed | gate acc | detect | Top-1 | Top-5 | coverage | precision | decode ms (mean/median) | identify ms (mean/median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inat | bird | 1046 | 0 | 97.1% | 99.3% | 42.1% | 55.9% | 39.4% | 73.8% | 1 / 1 | 154 / 149 |
| inat | mammal | 525 | 0 | 94.9% | 2.7% | 0.0% | 0.0% | 6.1% | 0.0% | 2 / 1 | 176 / 167 |

Definitions: Top-1/Top-5 use the highest-`score` box; coverage = best box at `level == species`; precision = Top-1 hit rate among those; failed images count as misses everywhere.

Top-1 hits gained by synonym normalisation of the truth (miss -> hit): inat/bird 8, inat/mammal 0

No box, by whole-frame gate class (none/person: the gate missed the animal; else the detector did): inat/bird bird 3, mammal 3, person 1; inat/mammal none 2, mammal 2, bird 1

## Confusion Top-10 — inat / bird

| truth | predicted | count |
|---|---|---|
| Branta canadensis | Branta hutchinsii | 15 |
| Egretta thula | Egretta garzetta | 13 |
| Dryobates nuttallii | Dryobates scalaris | 8 |
| Nannopterum auritum | Phalacrocorax capensis | 8 |
| Astur cooperii | Accipiter striatus | 6 |
| Aphelocoma californica | Aphelocoma woodhouseii | 6 |
| Nannopterum auritum | Phalacrocorax carbo | 6 |
| Tyto furcata | Tyto alba | 5 |
| Dryobates nuttallii | Leuconotopicus villosus | 5 |
| Buteo jamaicensis | Aquila chrysaetos | 4 |

## Confusion Top-10 — inat / mammal

| truth | predicted | count |
|---|---|---|
| Enhydra lutris | Melanitta perspicillata | 8 |
| Cervus canadensis | Struthio camelus | 7 |
| Odocoileus hemionus | Dromaius novaehollandiae | 6 |
| Sciurus griseus | Menura novaehollandiae | 6 |
| Sylvilagus audubonii | Eudromia formosa | 6 |
| Marmota flaviventris | Tinamotis pentlandii | 5 |
| Rangifer tarandus | Rhea pennata | 5 |
| Odocoileus hemionus | Ardeotis kori | 4 |
| Canis latrans | Ketupa coromanda | 4 |
| Lynx rufus | Bubo virginianus | 4 |
