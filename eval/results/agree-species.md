
## birds (n = 1046): each run against `teacher`, photo by photo

| run | same first name | same top-5 in order | same top-5 set | posterior of first name: max / mean abs diff | right -> wrong | wrong -> right |
|---|---|---|---|---|---|---|
| slim-fp16 | 1046 | 1046 | 1046 | 0.0000 / 0.00000 | 0 | 0 |
| slim-fp16-half | 1046 | 1017 | 1038 | 0.0630 / 0.00040 | 0 | 0 |
| int8 | 1045 | 890 | 976 | 0.1177 / 0.00183 | 0 | 1 |
| int8-half | 1045 | 882 | 974 | 0.1187 / 0.00183 | 0 | 1 |

## birds (n = 1046): each run against `teacher-no-geo`, photo by photo

| run | same first name | same top-5 in order | same top-5 set | posterior of first name: max / mean abs diff | right -> wrong | wrong -> right |
|---|---|---|---|---|---|---|
| slim-fp16-no-geo | 1046 | 1046 | 1046 | 0.0000 / 0.00000 | 0 | 0 |
| slim-fp16-half-no-geo | 1046 | 1007 | 1024 | 0.0597 / 0.00068 | 0 | 0 |
| int8-no-geo | 1044 | 891 | 987 | 0.1075 / 0.00310 | 1 | 1 |
| int8-half-no-geo | 1040 | 887 | 985 | 0.1225 / 0.00295 | 2 | 4 |

## mammals (n = 525): each run against `teacher`, photo by photo

| run | same first name | same top-5 in order | same top-5 set | posterior of first name: max / mean abs diff | right -> wrong | wrong -> right |
|---|---|---|---|---|---|---|
| slim-fp16 | 525 | 525 | 525 | 0.0000 / 0.00000 | 0 | 0 |
| slim-fp16-half | 525 | 506 | 518 | 0.0252 / 0.00055 | 0 | 0 |
| int8 | 523 | 441 | 493 | 0.0907 / 0.00273 | 0 | 1 |
| int8-half | 523 | 441 | 489 | 0.0955 / 0.00262 | 0 | 1 |

## mammals (n = 525): each run against `teacher-no-geo`, photo by photo

| run | same first name | same top-5 in order | same top-5 set | posterior of first name: max / mean abs diff | right -> wrong | wrong -> right |
|---|---|---|---|---|---|---|
| slim-fp16-no-geo | 525 | 525 | 525 | 0.0000 / 0.00000 | 0 | 0 |
| slim-fp16-half-no-geo | 525 | 508 | 516 | 0.0240 / 0.00050 | 0 | 0 |
| int8-no-geo | 524 | 437 | 491 | 0.1676 / 0.00370 | 0 | 1 |
| int8-half-no-geo | 523 | 428 | 487 | 0.1844 / 0.00357 | 0 | 1 |

(posterior differences over photos where both runs put the same name first)

## Rank above species: first name's genus / family is the truth's

| run | kind | n | species | genus | family (truth's family known) |
|---|---|---|---|---|---|
| teacher | bird | 1046 | 92.4 % | 94.9 % | 96.0 % (n = 1046) |
| teacher | mammal | 525 | 91.4 % | 94.3 % | 94.7 % (n = 525) |
| int8 | bird | 1046 | 92.4 % | 95.0 % | 96.0 % (n = 1046) |
| int8 | mammal | 525 | 91.6 % | 94.1 % | 94.5 % (n = 525) |
| int8-half | bird | 1046 | 92.4 % | 95.0 % | 96.0 % (n = 1046) |
| int8-half | mammal | 525 | 91.6 % | 94.1 % | 94.5 % (n = 525) |
| teacher-no-geo | bird | 1046 | 87.0 % | 94.3 % | 95.8 % (n = 1046) |
| teacher-no-geo | mammal | 525 | 85.7 % | 92.4 % | 93.7 % (n = 525) |
| int8-no-geo | bird | 1046 | 87.0 % | 94.4 % | 95.8 % (n = 1046) |
| int8-no-geo | mammal | 525 | 85.9 % | 92.4 % | 93.7 % (n = 525) |
| int8-half-no-geo | bird | 1046 | 87.2 % | 94.6 % | 95.8 % (n = 1046) |
| int8-half-no-geo | mammal | 525 | 85.9 % | 92.4 % | 93.7 % (n = 525) |
