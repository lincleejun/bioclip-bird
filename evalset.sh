#!/bin/zsh
# One evaluation run of any list in eval/ through gardener's pipeline, with distill/serve.py's swaps.
#   ./evalset.sh OUT LIST [serve.py args ...] [-- gardener-engine eval args]
#   ./evalset.sh runs/e2-teacher eval/inat2026.csv                                  # original models
#   ./evalset.sh runs/e2-int8-half eval/inat2026.csv --slim models/bioclip-2.5-vith14-visual-int8.safetensors --half
#   ./evalset.sh runs/e2-teacher-no-geo eval/inat2026.csv -- --no-geo
# LIST paths are relative to eval/; a copy with absolute paths is made in OUT. -> OUT/report.md, OUT/preds.ndjson
cd "$(dirname "$0")"; OUT=$PWD/$1; LIST=$PWD/$2; shift 2; B=$(cd ../bioscan && pwd)
SERVE=(); while (( $# )) && [[ $1 != -- ]]; do SERVE+=$1; shift; done; [[ $1 == -- ]] && shift
mkdir -p $OUT
python3 -c "import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); w=csv.DictWriter(open(sys.argv[2],'w',newline=''),r[0].keys()); w.writeheader(); [w.writerow({**x,'path':sys.argv[3]+'/'+x['path']}) for x in r]" $LIST $OUT/list.csv $PWD/eval
curl -s -m 2 http://127.0.0.1:8771/health >/dev/null && { echo "port 8771 is busy"; exit 1; }
PYTHONUNBUFFERED=1 ./run distill/serve.py --port 8771 $SERVE --allow-root $PWD/eval --allow-root $OUT > $OUT/serve.log 2>&1 &
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8771/health >/dev/null && break; sleep 2; done
(cd $B && uv run -q gardener-engine --url http://127.0.0.1:8771 eval $OUT/list.csv --out $OUT "$@" > $OUT/eval.log 2>&1; echo "exit $?" >> $OUT/eval.log)
pkill -f "distill/serve.py --port 8771"; sleep 3
grep -E "^\| [a-z0-9-]+ \| bird " $OUT/report.md
