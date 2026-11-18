## Bird photos (n = 1046): each detector's best box against OWLv2's (run `teacher`); no true boxes exist

| detector | run | photos with a box | boxes per photo (mean) | best-box IoU with OWLv2: mean / >= 0.5 / >= 0.9 (both have a box) | best box's share of the frame: p10 / median / p90 |
|---|---|---|---|---|---|
| OWLv2 | `teacher` | 1039 | 1.49 | 1.000 / 100.0 % / 100.0 % (n = 1039) | 0.006 / 0.075 / 0.375 |
| OWLv2 int8 | `det-owlv2-int8` | 1039 | 1.47 | 0.971 / 98.7 % / 98.3 % (n = 1039) | 0.006 / 0.075 / 0.375 |
| RT-DETRv2 r50 | `det-rtdetr-r50` | 1032 | 1.87 | 0.833 / 88.9 % / 70.2 % (n = 1027) | 0.007 / 0.076 / 0.374 |
| RT-DETRv2 r18 | `det-rtdetr-r18` | 1026 | 2.20 | 0.806 / 86.9 % / 64.3 % (n = 1022) | 0.006 / 0.075 / 0.374 |
| D-FINE small | `det-dfine-small` | 1026 | 2.28 | 0.809 / 86.9 % / 66.3 % (n = 1022) | 0.007 / 0.075 / 0.371 |
| D-FINE nano | `det-dfine-nano` | 1027 | 2.35 | 0.787 / 85.4 % / 60.3 % (n = 1023) | 0.007 / 0.075 / 0.387 |
| YOLOv10 m | `det-yolov10m` | 956 | 1.23 | 0.842 / 89.7 % / 72.7 % (n = 956) | 0.008 / 0.080 / 0.373 |
| YOLOv10 s | `det-yolov10s` | 932 | 1.22 | 0.820 / 88.1 % / 68.2 % (n = 932) | 0.009 / 0.081 / 0.379 |
| YOLOv10 n | `det-yolov10n` | 864 | 1.08 | 0.802 / 87.2 % / 60.5 % (n = 864) | 0.010 / 0.090 / 0.384 |

Boxes per photo counts every box the engine kept (any kind); the COCO detectors report birds only.
