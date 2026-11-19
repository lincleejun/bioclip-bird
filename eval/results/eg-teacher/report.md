# gardener-engine eval report

- groundtruth: eval/inat-global.csv (absolute-path copy)
- preds: preds.ndjson
- images: 2632
- geo: True
- preds schema: 1
- complete: True
- synonyms: data/names/synonyms.csv (gardener)
- generated: 2026-11-18 21:31:15
- wall_s: 651.1
- engine: {"detail_edge": 3072, "models": {"detect": "google/owlv2-base-patch16-ensemble@cfd3195ba4ea", "gate": "google/siglip2-base-patch16-224@75de2d55ec2d", "geo": "birdnet-geo-3.0", "label_maps": {"bird": "avilist_map.csv@55d1aa841f8d", "mammal": "mdd_map.csv@758a82fafbbd"}, "names": {"bird": "avilist-2026@a4f7858b14c15128", "mammal": "mdd-2025@3eab033f008ac62f", "other_animal": "tol200m-animalia@4674544e7661da91"}, "priors": {"bird": "birdnet-geo-3.0", "mammal": "birdnet-geo-3.0", "other_animal": null}, "species": "imageomics/bioclip-2.5-vith14@6e3d04e3d652", "taxonomy": {"bird": "AviList v2026", "mammal": "MDD v2.5", "other_animal": "TreeOfLife-200M @ 5f2dc493"}}, "settings": "79a2ce363c57", "version": "0.1.0"}

## Metrics by tier × kind

| tier | kind | n | failed | gate acc | detect | Top-1 | Top-5 | coverage | precision | decode ms (mean/median) | identify ms (mean/median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inat-global | bird | 2632 | 0 | 97.8% | 99.3% | 77.2% | 92.3% | 77.5% | 88.5% | 3 / 2 | 239 / 205 |

Definitions: Top-1/Top-5 use the highest-`score` box; coverage = best box at `level == species`; precision = Top-1 hit rate among those; failed images count as misses everywhere.

Top-1 hits gained by synonym normalisation of the truth (miss -> hit): inat-global/bird 0

No box, by whole-frame gate class (none/person: the gate missed the animal; else the detector did): inat-global/bird bird 10, none 4, other_animal 3, mammal 1

## Confusion Top-10 — inat-global / bird

| truth | predicted | count |
|---|---|---|
| Drymophila rubricollis | Drymophila ferruginea | 3 |
| Microtarsus fuscoflavescens | Microtarsus melanocephalos | 3 |
| Pitta forsteni | Pitta sordida | 3 |
| Philohydor lictor | Pitangus sulphuratus | 3 |
| Eriocnemis nigrivestis | Eriocnemis vestita | 3 |
| Pachyptila crassirostris | Pachyptila turtur | 3 |
| Cincloramphus mariae | Elaphrornis palliseri | 3 |
| Cacatua pastinator | Cacatua sanguinea | 3 |
| Malacoptila fusca | Malacoptila mystacalis | 3 |
| Cinnyris frenatus | Cinnyris jugularis | 3 |
