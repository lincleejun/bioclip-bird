#!/bin/zsh
# A student in the teacher's seat, scored on gardener's golden set (same detector, crops, geo prior, scoring).
#   ./eval_student.sh runs/s-small-1/student.pt            # -> runs/s-small-1/golden/report.md
#   ./eval_student.sh runs/s-small-1/student.pt --no-geo   # -> runs/s-small-1/golden-no-geo/report.md
# Teacher on the same set today: birds top-1 92.4 %, top-5 96.9 % (runs/teacher-golden/report.md).
cd "$(dirname "$0")"; CK=$1; shift; B=$(cd ../bioscan && pwd); OUT=$PWD/$(dirname $CK)/golden${1:+-no-geo}; mkdir -p $OUT
curl -s -m 2 http://127.0.0.1:8771/health >/dev/null && { echo "port 8771 is busy"; exit 1; }
PYTHONUNBUFFERED=1 ./run distill/serve.py --port 8771 --student $CK --allow-root $B/data --allow-root $OUT > $OUT/serve.log 2>&1 &
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8771/health >/dev/null && break; sleep 2; done
(cd $B && uv run -q gardener-engine --url http://127.0.0.1:8771 eval data/inat/groundtruth-inat.csv --out $OUT "$@" > $OUT/eval.log 2>&1; echo "exit $?" >> $OUT/eval.log)
pkill -f "distill/serve.py --port 8771"
sed -n 16,19p $OUT/report.md
