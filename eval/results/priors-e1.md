# Location priors on the same bird crops (groundtruth.csv, boxes of run `teacher`)

1039 bird crops of 1046 bird photos (7 without a box are left out), 42 species. Regions: {'USA.5_1': 1039}; 0 crops fell back to their country (state not matched). Own records taken out: 1039.

True names with no GBIF record in their region in any month (own records out): 24 crops, *Bubo virginianus* 24.

| prior | all (n = 1039) top-1 / top-5 |
|---|---|
| no location | 87.9 % / 96.9 % |
| BirdNET geo (gardener) | 92.1 % / 97.9 % |
| GBIF soft, beta 0.5 | 94.4 % / 98.2 % |
| GBIF soft, beta 1.0 | 95.5 % / 98.8 % |
| GBIF hard list | 92.5 % / 96.2 % |
| GBIF soft, beta 1.0, own records left in | 95.5 % / 98.8 % |
| GBIF hard list, own records left in | 92.5 % / 96.2 % |
