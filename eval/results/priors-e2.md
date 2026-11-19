# Location priors on the same bird crops (inat2026.csv, boxes of run `e2-teacher`)

632 bird crops of 637 bird photos (5 without a box are left out), 228 species. Regions: {'USA.5_1': 94, 'CAN.2_1': 51, 'USA.44_1': 44, 'USA.3_1': 42, 'CAN.9_1': 39, 'USA.10_1': 31, 'CHN.6_1': 26, 'CAN.11_1': 20} ...; 0 crops at country level (no state/province table or name match), 0 with no known country (no GBIF prior). Own records taken out: 632. GBIF table: data/rarity.

True names with no GBIF record in their region in any month (own records out): 29 crops, *Trogon ambiguus* 3, *Lophospiza trivirgata* 3, *Cistothorus palustris* 3, *Anarhynchus montanus* 3, *Anarhynchus nivosus* 3, *Bubo virginianus* 3, *Enicurus sinensis* 3, *Larus mongolicus* 2, *Larus vegae* 1, *Phyllergates cucullatus* 1, *Tachyspiza soloensis* 1, *Vireo plumbeus* 1, *Phylloscopus intermedius* 1, *Terpsiphone incei* 1.

| prior | all (n = 632) top-1 / top-5 | country CA (n = 156) top-1 / top-5 | country CN (n = 150) top-1 / top-5 | country US (n = 326) top-1 / top-5 | species: least observed third (n = 211) top-1 / top-5 | species: middle third (n = 212) top-1 / top-5 | species: most observed third (n = 209) top-1 / top-5 |
|---|---|---|---|---|---|---|---|
| no location | 79.1 % / 91.6 % | 84.0 % / 96.8 % | 67.3 % / 82.0 % | 82.2 % / 93.6 % | 64.0 % / 83.4 % | 84.9 % / 94.3 % | 88.5 % / 97.1 % |
| BirdNET geo (gardener) | 83.7 % / 93.4 % | 89.1 % / 96.2 % | 74.0 % / 87.3 % | 85.6 % / 94.8 % | 71.6 % / 86.7 % | 86.3 % / 95.8 % | 93.3 % / 97.6 % |
| GBIF soft, beta 0.5 | 82.9 % / 93.8 % | 89.7 % / 96.8 % | 69.3 % / 86.0 % | 85.9 % / 96.0 % | 68.7 % / 86.3 % | 86.8 % / 97.2 % | 93.3 % / 98.1 % |
| GBIF soft, beta 1.0 | 83.9 % / 94.3 % | 90.4 % / 96.8 % | 74.0 % / 87.3 % | 85.3 % / 96.3 % | 70.1 % / 86.7 % | 87.3 % / 97.6 % | 94.3 % / 98.6 % |
| GBIF hard list | 82.9 % / 92.7 % | 85.3 % / 92.9 % | 74.7 % / 86.7 % | 85.6 % / 95.4 % | 72.0 % / 86.7 % | 87.3 % / 96.7 % | 89.5 % / 94.7 % |
| GBIF soft, beta 1.0, own records left in | 83.9 % / 94.3 % | 90.4 % / 96.8 % | 74.0 % / 87.3 % | 85.3 % / 96.3 % | 70.1 % / 86.7 % | 87.3 % / 97.6 % | 94.3 % / 98.6 % |
| GBIF hard list, own records left in | 83.5 % / 93.4 % | 85.3 % / 92.9 % | 76.7 % / 88.7 % | 85.9 % / 95.7 % | 73.0 % / 87.7 % | 87.3 % / 96.7 % | 90.4 % / 95.7 % |
