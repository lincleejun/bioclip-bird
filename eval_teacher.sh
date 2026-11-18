#!/bin/zsh
# Teacher through the swapped-encoder service: records every species crop + vector, scores the golden set.
# Reproducing gardener's baseline (birds top-1 0.9168, baselines/golden-inat-2026-11-13.json) is the harness check.
cd "$(dirname "$0")"; B=../bioscan; OUT=$PWD/runs/teacher-golden; REC=$PWD/data/rec/golden; mkdir -p $OUT $REC
PYTHONUNBUFFERED=1 ./run distill/serve.py --port 8771 --record $REC --allow-root $(cd $B && pwd)/data --allow-root $OUT > $OUT/serve.log 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8771/health >/dev/null && break; sleep 2; done
(cd $B && uv run -q gardener-engine --url http://127.0.0.1:8771 eval data/inat/groundtruth-inat.csv --out $OUT > $OUT/eval.log 2>&1; echo "exit $?" >> $OUT/eval.log)
kill $SP; pkill -f "distill/serve.py --port 8771"
