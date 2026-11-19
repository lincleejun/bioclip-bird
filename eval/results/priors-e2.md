# Location priors on the same bird crops (inat2026.csv, boxes of run `e2-teacher`)

632 bird crops of 637 bird photos (5 without a box are left out), 228 species. Regions: {'USA.5_1': 94, 'CAN.2_1': 51, 'USA.44_1': 44, 'USA.3_1': 42, 'CAN.9_1': 39, 'USA.10_1': 31, 'CHN.6_1': 26, 'CAN.11_1': 20} .... Truth names on no GBIF list of their region (any month): 25.

| prior | all (n = 632) top-1 / top-5 | country CA (n = 156) top-1 / top-5 | country CN (n = 150) top-1 / top-5 | country US (n = 326) top-1 / top-5 | species: least observed third (n = 211) top-1 / top-5 | species: middle third (n = 212) top-1 / top-5 | species: most observed third (n = 209) top-1 / top-5 |
|---|---|---|---|---|---|---|---|
| no location | 79.1 % / 91.6 % | 84.0 % / 96.8 % | 67.3 % / 82.0 % | 82.2 % / 93.6 % | 64.0 % / 83.4 % | 84.9 % / 94.3 % | 88.5 % / 97.1 % |
| BirdNET geo (gardener) | 83.7 % / 93.4 % | 89.1 % / 96.2 % | 74.0 % / 87.3 % | 85.6 % / 94.8 % | 71.6 % / 86.7 % | 86.3 % / 95.8 % | 93.3 % / 97.6 % |
| GBIF soft, beta 0.5 | 83.1 % / 93.8 % | 89.7 % / 96.8 % | 70.0 % / 86.0 % | 85.9 % / 96.0 % | 68.7 % / 86.3 % | 87.3 % / 97.2 % | 93.3 % / 98.1 % |
| GBIF soft, beta 1.0 | 83.9 % / 94.3 % | 90.4 % / 96.8 % | 74.0 % / 87.3 % | 85.3 % / 96.3 % | 70.1 % / 86.7 % | 87.3 % / 97.6 % | 94.3 % / 98.6 % |
| GBIF hard list | 83.5 % / 93.8 % | 85.3 % / 92.9 % | 76.7 % / 90.7 % | 85.9 % / 95.7 % | 73.0 % / 89.1 % | 87.3 % / 96.7 % | 90.4 % / 95.7 % |
