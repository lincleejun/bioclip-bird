# Speed, load time and memory (Apple M4 Max, 64 GB)

Raw lines: `raw.jsonl`; method: `eval/speed.py`'s docstring. Machine at the start of each batch: load average [5.36, 19.06, 29.74]; load average [11.55, 11.3, 20.73]; load average [8.29, 10.46, 19.68].

## Species model (image tower, random 224 px input, no preprocessing)

| model | device | compute | load s | batch 1 ms median / p90 | batch 32 images/s | peak RSS GB | MPS driver GB |
|---|---|---|---|---|---|---|---|
| original | mps | fp32 | 8.9 | 46.9 / 47.6 | 32.0 | 4.35 | 4.47 |
| original | mps | fp16 | 6.2 | 41.4 / 42.8 | 38.6 | 4.34 | 4.4 |
| original | cpu | fp32 | 5.8 | 197.4 / 213.5 | 7.6 | 8.29 | - |
| fp16 | mps | fp32 | 6.1 | 46.3 / 46.9 | 32.8 | 5.18 | 4.47 |
| fp16 | mps | fp16 | 5.9 | 41.7 / 43.2 | 38.8 | 5.19 | 2.26 |
| fp16 | cpu | fp32 | 5.7 | 199.0 / 209.0 | 7.6 | 5.19 | - |
| int8 | mps | fp32 | 6.3 | 47.1 / 48.1 | 32.1 | 7.24 | 4.47 |
| int8 | mps | fp16 | 5.8 | 41.8 / 42.9 | 38.8 | 7.24 | 2.26 |
| int8 | cpu | fp32 | 5.8 | 193.7 / 200.8 | 7.7 | 7.24 | - |
| original | mps | fp32 | 6.2 | 46.8 / 48.5 | 32.0 | 4.35 | 4.47 |
| original | mps | fp16 | 6.1 | 42.0 / 43.0 | 38.7 | 4.35 | 4.4 |
| original | cpu | fp32 | 5.7 | 190.6 / 200.9 | 7.6 | 8.28 | - |
| fp16 | mps | fp32 | 5.6 | 46.4 / 46.9 | 32.2 | 5.19 | 4.47 |
| fp16 | mps | fp16 | 5.7 | 41.1 / 41.4 | 39.0 | 5.18 | 2.26 |
| fp16 | cpu | fp32 | 5.6 | 195.5 / 208.4 | 7.5 | 5.18 | - |
| int8 | mps | fp32 | 6.0 | 47.3 / 47.9 | 32.7 | 7.24 | 4.47 |
| int8 | mps | fp16 | 5.9 | 41.2 / 41.7 | 39.3 | 7.24 | 2.26 |
| int8 | cpu | fp32 | 5.7 | 187.6 / 201.7 | 7.1 | 7.24 | - |

## Detector (100 bird photos, whole photo in, bird boxes out)

| detector | device | load s | ms per photo median / p90 | peak RSS GB | MPS driver GB |
|---|---|---|---|---|---|
| owlv2 | mps | 2.5 | 131.5 / 135.6 | 0.67 | 1.14 |
| owlv2 | cpu | 2.0 | 512.8 / 549.7 | 1.28 | - |
| owlv2-int8 | mps | 4.6 | 132.5 / 137.7 | 1.99 | 1.14 |
| owlv2-int8 | cpu | 4.5 | 518.7 / 552.3 | 1.99 | - |
| rtdetr-r50 | mps | 2.2 | 37.4 / 42.4 | 0.67 | 1.3 |
| rtdetr-r50 | cpu | 2.0 | 122.7 / 136.1 | 1.14 | - |
| rtdetr-r18 | mps | 2.1 | 20.1 / 32.7 | 0.66 | 1.21 |
| rtdetr-r18 | cpu | 1.9 | 68.8 / 76.8 | 1.0 | - |
| dfine-small | mps | 2.2 | 20.3 / 30.4 | 0.68 | 1.2 |
| dfine-small | cpu | 1.9 | 142.5 / 168.9 | 0.73 | - |
| dfine-nano | mps | 2.1 | 18.2 / 29.5 | 0.68 | 0.16 |
| dfine-nano | cpu | 2.0 | 113.5 / 132.1 | 0.65 | - |
| yolov10m | cpu | 0.5 | 53.9 / 63.5 | 0.7 | - |
| yolov10s | cpu | 0.5 | 24.3 / 26.5 | 0.56 | - |
| yolov10n | cpu | 0.5 | 13.7 / 14.5 | 0.45 | - |
| owlv2 | mps | 2.1 | 129.7 / 133.5 | 0.67 | 1.14 |
| owlv2 | cpu | 2.0 | 486.8 / 517.2 | 1.28 | - |
| owlv2-int8 | mps | 4.5 | 130.4 / 135.5 | 1.99 | 1.14 |
| owlv2-int8 | cpu | 4.5 | 503.8 / 537.8 | 1.99 | - |
| rtdetr-r50 | mps | 2.1 | 37.3 / 43.4 | 0.66 | 1.31 |
| rtdetr-r50 | cpu | 2.0 | 117.1 / 128.7 | 1.14 | - |
| rtdetr-r18 | mps | 2.0 | 24.0 / 37.3 | 0.66 | 1.21 |
| rtdetr-r18 | cpu | 1.9 | 66.4 / 72.5 | 1.0 | - |
| dfine-small | mps | 2.0 | 19.3 / 31.7 | 0.69 | 1.2 |
| dfine-small | cpu | 2.1 | 142.5 / 165.5 | 0.73 | - |
| dfine-nano | mps | 2.1 | 16.0 / 25.8 | 0.68 | 0.16 |
| dfine-nano | cpu | 1.8 | 108.4 / 125.5 | 0.65 | - |
| yolov10m | cpu | 0.5 | 52.8 / 58.6 | 0.7 | - |
| yolov10s | cpu | 0.5 | 23.9 / 25.2 | 0.56 | - |
| yolov10n | cpu | 0.5 | 13.5 / 14.5 | 0.45 | - |

## Core ML conversions on the Mac (not a phone; batch 1, model call only)

| model | compute units | load s (includes compiling) | ms median / p90 | outputs all zero | peak RSS GB |
|---|---|---|---|---|---|
| BioCLIP | CPU_ONLY | 2.4 | 99.3 / 105.3 | no | 2.36 |
| BioCLIP | CPU_AND_GPU | 2.3 | 415.6 / 423.4 | no | 1.72 |
| BioCLIP | CPU_AND_NE | 9.3 | 71.3 / 71.7 | no | 1.04 |
| BioCLIP | ALL | 9.3 | 71.3 / 72.6 | no | 1.7 |
| Detector | CPU_ONLY | 0.3 | 16.5 / 17.9 | no | 0.44 |
| Detector | CPU_AND_GPU | 0.3 | 4.0 / 4.1 | no | 0.46 |
| Detector | CPU_AND_NE | 0.4 | 15.9 / 17.9 | no | 0.44 |
| Detector | ALL | 0.4 | 4.1 / 4.2 | no | 0.47 |
