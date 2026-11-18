# Species model alone on bird crops

1039 bird crops of the evaluation set (the original pipeline's best box on each of the 1,046 bird photos that got one), 11,172 AviList names, image only (no location prior). Device: mps.

| model | top-1 | top-5 | cosine to the original's vector: mean / min | same first name as the original |
|---|---|---|---|---|
| BioCLIP 2.5 original (fp32) | 87.8 % | 96.9 % | 1.00000 / 1.00000 | 1039 of 1039 |
| BioCLIP 2.5 fp16 file, fp32 compute | 87.8 % | 96.9 % | 1.00000 / 1.00000 | 1039 of 1039 |
| BioCLIP 2.5 fp16 file, fp16 compute | 87.8 % | 96.9 % | 0.99999 / 0.99977 | 1039 of 1039 |
| BioCLIP 2.5 int8 file, fp32 compute | 87.8 % | 96.9 % | 0.99983 / 0.98226 | 1037 of 1039 |
| BioCLIP 2.5 int8 file, fp16 compute | 87.9 % | 96.9 % | 0.99982 / 0.98222 | 1034 of 1039 |
| BioCLIP 1 (ViT-B/16), own text tower | 51.4 % | 74.8 % | - (other vector space) | - |

## Against the vectors the original recorded in its pipeline run (3171 crops, birds and mammals, saved as JPEG)

| model | cosine: mean / min | same first bird name (of the 11,172) |
|---|---|---|
| BioCLIP 2.5 original (fp32) | 0.99118 / 0.78823 | 2882 of 3171 |
| BioCLIP 2.5 fp16 file, fp32 compute | 0.99118 / 0.78823 | 2882 of 3171 |
| BioCLIP 2.5 fp16 file, fp16 compute | 0.99118 / 0.78967 | 2881 of 3171 |
| BioCLIP 2.5 int8 file, fp32 compute | 0.99105 / 0.79057 | 2871 of 3171 |
| BioCLIP 2.5 int8 file, fp16 compute | 0.99106 / 0.79071 | 2871 of 3171 |
