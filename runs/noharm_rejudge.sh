#!/usr/bin/env bash
# The same plans, judged again under a new name: by another judge family, or by the same judge for
# its own noise. usage: noharm_rejudge.sh <source-arm> <new-name> <judge> [cases]
cd "$(dirname "$0")/.."
src=$1; name=$2; judge=$3; cases=${4:-}
P=sources/mast/benchmarks/donoharm/.venv/bin/python
D=sources/mast/results/raw/donoharm/default
[ -f "$D/$name.jsonl" ] || cp "$D/$src.jsonl" "$D/$name.jsonl"
copt=(); [ -n "$cases" ] && copt=(--cases "$cases")
threads=3; [[ "$judge" == claude* ]] && threads=6; [[ "$judge" == grok* ]] && threads=2
for attempt in 1 2 3; do
  $P scripts/noharm_kit.py score --name "$name" --judge "$judge" --threads $threads "${copt[@]}" > tmp/runs/noharm_score_$name.log 2>&1
  grep -q "Overall *F1_weighted" tmp/runs/noharm_score_$name.log && break
  sleep 300
done
echo "REJUDGED $src as $name by $judge: $(grep -m1 -E 'Overall *F1_weighted' tmp/runs/noharm_score_$name.log) $(date -u +%H:%M)"
