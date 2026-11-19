# gardener-engine eval report

- groundtruth: eval/inat2026.csv (absolute-path copy)
- preds: preds.ndjson
- images: 637
- geo: True
- preds schema: 1
- complete: True
- synonyms: data/names/synonyms.csv (gardener)
- generated: 2026-11-18 19:13:31
- wall_s: 139.9
- engine: {"detail_edge": 3072, "models": {"detect": "google/owlv2-base-patch16-ensemble@cfd3195ba4ea", "gate": "google/siglip2-base-patch16-224@75de2d55ec2d", "geo": "birdnet-geo-3.0", "label_maps": {"bird": "avilist_map.csv@55d1aa841f8d", "mammal": "mdd_map.csv@758a82fafbbd"}, "names": {"bird": "avilist-2026@a4f7858b14c15128", "mammal": "mdd-2025@3eab033f008ac62f", "other_animal": "tol200m-animalia@4674544e7661da91"}, "priors": {"bird": "birdnet-geo-3.0", "mammal": "birdnet-geo-3.0", "other_animal": null}, "species": "imageomics/bioclip-2.5-vith14@6e3d04e3d652", "taxonomy": {"bird": "AviList v2026", "mammal": "MDD v2.5", "other_animal": "TreeOfLife-200M @ 5f2dc493"}}, "settings": "79a2ce363c57", "version": "0.1.0"}

## Metrics by tier × kind

| tier | kind | n | failed | gate acc | detect | Top-1 | Top-5 | coverage | precision | decode ms (mean/median) | identify ms (mean/median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inat2026 | bird | 637 | 0 | 96.2% | 99.1% | 82.3% | 92.8% | 84.1% | 91.4% | 1 / 1 | 201 / 200 |

Definitions: Top-1/Top-5 use the highest-`score` box; coverage = best box at `level == species`; precision = Top-1 hit rate among those; failed images count as misses everywhere.

Top-1 hits gained by synonym normalisation of the truth (miss -> hit): inat2026/bird 0

No box, by whole-frame gate class (none/person: the gate missed the animal; else the detector did): inat2026/bird none 3, bird 1, mammal 1

## Confusion Top-10 — inat2026 / bird

| truth | predicted | count |
|---|---|---|
| Trogon ambiguus | Trogon elegans | 3 |
| Anser fabalis | Anser albifrons | 3 |
| Tachyspiza soloensis | Butastur indicus | 3 |
| Calidris pugnax | Calidris alpina | 2 |
| Cygnus cygnus | Cygnus olor | 2 |
| Turdus naumanni | Turdus eunomus | 2 |
| Amazona autumnalis | Amazona viridigenalis | 2 |
| Toxostoma bendirei | Toxostoma curvirostre | 2 |
| Leucosticte australis | Leucosticte tephrocotis | 2 |
| Psittacara mitratus | Psittacara erythrogenys | 2 |
