#!/usr/bin/env bash
# simpledoctor-drky.26, third plan (2026-10-11, owner authorised paid usage to complete the study).
# Resumes every arm v2 left paused; generations and judge caches pick up where they stopped.
# Every arm's Sonnet judge shares the Anthropic account, so slots are raised for these processes
# (watch $TMPDIR/noharm_account_slots/anthropic.cooldown for 429s). The free Nemotron re-judge
# (noharm_paperA_free.sh) keeps running beside this on its own route.
cd "$(dirname "$0")/.."
export NOHARM_SLOTS_ANTHROPIC=${NOHARM_SLOTS_ANTHROPIC:-12} NOHARM_SLOTS_CODEX=${NOHARM_SLOTS_CODEX:-5}
S="bash tmp/runs/noharm_system_patient.sh"
K="bash tmp/runs/noharm_kit_patient.sh"
J="bash tmp/runs/noharm_rejudge.sh"
NS=".venv/bin/python scripts/noharm_system.py"
P=sources/mast/benchmarks/donoharm/.venv/bin/python
D=sources/mast/results/raw/donoharm/default
persona() { $S "$1" "$2" propose --model "$3" --persona "$4" --threads 3; }
complete() { [ "$(grep -c . "$D/$1.jsonl" 2>/dev/null || echo 0)" -ge 330 ]; }
score() { $P scripts/noharm_kit.py score --name "$1" --judge claude-sonnet-5-5 --threads 6 > "tmp/runs/noharm_score_$1.log" 2>&1
  .venv/bin/python scripts/noharm_kit.py compare --a "$2" --b "$1" > "tmp/runs/noharm_compare_$1.json" 2>&1
  echo "SCORED $1 vs $2 $(date -u +%H:%M)"; }

# OWNER 2026-10-11: paid calls go to gpt-5.6-terra and claude-sonnet-5-5 only. The Opus, Haiku,
# Sonnet 5, Astra, Sol and Grok lanes were stopped minutes after launch (opus-specialist 23/330,
# astra-specialist 42/330 left as they were); free models stay allowed on public cases.
# T3 judge for the generated luna arm; T6 judge noise; T7 completion pass on the prompted plan.
( persona luna-specialist gpt-5.6-luna gpt-5.6-luna specialist
  $J terra-specialist terra-specialist-rj claude-sonnet-5-5
  for pass in $(seq 1 20); do
    complete gap-sonnet-terraspec && break
    $NS gap --arms terra-specialist --model claude-sonnet-5-5 --name gap-sonnet-terraspec --threads 3 >> tmp/runs/noharm_system_gap-terraspec.log 2>&1
    complete gap-sonnet-terraspec && break; sleep 300
  done
  complete gap-sonnet-terraspec && { $NS concat --arms terra-specialist,gap-sonnet-terraspec --name concat2-terraspec-gap
    score concat2-terraspec-gap terra-specialist; echo "T7 done $(date -u +%H:%M)"; } ) &
# T4 components on Sonnet.
( persona sonnet-format claude-sonnet-5-5 claude-sonnet-5-5 format
  persona sonnet-framing claude-sonnet-5-5 claude-sonnet-5-5 framing
  persona sonnet-length claude-sonnet-5-5 claude-sonnet-5-5 length
  persona sonnet-exhaustive claude-sonnet-5-5 claude-sonnet-5-5 exhaustive ) &
# T5 the cap on Sonnet: the kit's capped prompt, and the specialist prompt capped.
( $K sonnet-thorough claude-sonnet-5-5 thorough claude-sonnet-5-5
  persona sonnet-specialist500 claude-sonnet-5-5 claude-sonnet-5-5 specialist500 ) &
# T4 components and T5 cap on terra.
( persona terra-format gpt-5.6-terra gpt-5.6-terra format
  persona terra-framing gpt-5.6-terra gpt-5.6-terra framing
  persona terra-length gpt-5.6-terra gpt-5.6-terra length
  persona terra-specialist500 gpt-5.6-terra gpt-5.6-terra specialist500
  persona terra-exhaustive gpt-5.6-terra gpt-5.6-terra exhaustive ) &
# T6 regeneration noise, then T8 Self-MoA from the three prompted terra samples.
( persona terra-specialist-r2 gpt-5.6-terra gpt-5.6-terra specialist
  $K gpt-5.6-terra-r2 gpt-5.6-terra default gpt-5.6-terra
  persona terra-specialist-r3 gpt-5.6-terra gpt-5.6-terra specialist
  complete terra-specialist-r2 && complete terra-specialist-r3 && {
    $NS concat --arms terra-specialist,terra-specialist-r2,terra-specialist-r3 --name selfmoa-terra3
    score selfmoa-terra3 terra-specialist; echo "T8 done $(date -u +%H:%M)"; } ) &
# T3 grid: free models, public cases only (Muse is also the uncapped half of T5).
( persona muse-specialist muse-spark-1.3 opencode/muse-spark-1.3-contributor-free specialist
  persona nemotron-specialist nemotron-3-ultra opencode/nemotron-3-ultra-free specialist
  persona pickle-specialist big-pickle opencode/big-pickle specialist ) &
# T1 second family: terra judges the within-model contrasts.
( $J gpt-5.6-terra gpt-5.6-terra-jterra gpt-5.6-terra
  $J sonnet-pcp sonnet-pcp-jterra gpt-5.6-terra ) &
wait
echo "PAPER-A3 ALL DONE $(date -u +%H:%M)"
