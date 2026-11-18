#!/bin/zsh
# Detector bench: each detector of distill/detectors.py in OWLv2's seat, scored on gardener's golden set with everything
# else unchanged (SigLIP2 gates, original teacher, geo prior).   ./detbench.sh NAME[@THRESHOLD] ...   -> runs/det-NAME[-tT]/golden
cd "$(dirname "$0")"; B=$(cd ../bioscan && pwd)
for SPEC in "$@"; do
  NAME=${SPEC%@*}; T=${${SPEC#*@}:#$SPEC}; OUT=$PWD/runs/det-$NAME${T:+-t$T}/golden; mkdir -p $OUT; rm -f $OUT/boxes.jsonl
  while curl -s -m 2 http://127.0.0.1:8771/health >/dev/null; do sleep 3; done
  PYTHONUNBUFFERED=1 ./run distill/serve.py --port 8771 --detector $NAME ${T:+--det-threshold=$T} --det-log $OUT/boxes.jsonl --allow-root $B/data --allow-root $OUT > $OUT/serve.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8771/health >/dev/null && break; sleep 2; done
  (cd $B && uv run -q gardener-engine --url http://127.0.0.1:8771 eval data/inat/groundtruth-inat.csv --out $OUT > $OUT/eval.log 2>&1; echo "exit $?" >> $OUT/eval.log)
  pkill -f "distill/serve.py --port 8771"; sleep 3
  echo "$SPEC $(sed -n 18p $OUT/report.md | cut -c1-110)"
done
