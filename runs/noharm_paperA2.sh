#!/usr/bin/env bash
# simpledoctor-drky.26, second plan: v1 (noharm_paperA.sh) ran at ~40 items/hour per arm because
# every arm's judge shares the Anthropic account's 6 slots. Claim-critical arms first, the rest in
# a final lane; slots raised for these processes (watch anthropic.cooldown for 429s).
cd "$(dirname "$0")/.."
export NOHARM_SLOTS_ANTHROPIC=${NOHARM_SLOTS_ANTHROPIC:-9} NOHARM_SLOTS_CODEX=${NOHARM_SLOTS_CODEX:-4}
S="bash tmp/runs/noharm_system_patient.sh"
K="bash tmp/runs/noharm_kit_patient.sh"
J="bash tmp/runs/noharm_rejudge.sh"
NS=".venv/bin/python scripts/noharm_system.py"
P=sources/mast/benchmarks/donoharm/.venv/bin/python
D=sources/mast/results/raw/donoharm/default
persona() { $S "$1" "$2" propose --model "$3" --persona "$4" --threads 3; }
complete() { [ "$(grep -c . "$D/$1.jsonl" 2>/dev/null || echo 0)" -ge 330 ]; }

# Grid (T3): judge the two already generated, then the Anthropic models.
( persona sonnet-specialist claude-sonnet-5-5 claude-sonnet-5-5 specialist
  persona luna-specialist gpt-5.6-luna gpt-5.6-luna specialist
  persona opus-specialist claude-opus-5-5 claude-opus-5-5 specialist
  persona haiku-specialist claude-haiku-4-5-20251001 claude-haiku-4-5-20251001 specialist
  persona sonnet5-specialist claude-sonnet-5 claude-sonnet-5 specialist ) &
# Components on Sonnet (T4) and the kit's capped prompt on Sonnet (T5).
( persona sonnet-format claude-sonnet-5-5 claude-sonnet-5-5 format
  persona sonnet-framing claude-sonnet-5-5 claude-sonnet-5-5 framing
  persona sonnet-length claude-sonnet-5-5 claude-sonnet-5-5 length
  $K sonnet-thorough claude-sonnet-5-5 thorough claude-sonnet-5-5 ) &
# Judge noise (T6) and a completion pass on the prompted plan (T7).
( $J terra-specialist terra-specialist-rj claude-sonnet-5-5
  for pass in $(seq 1 20); do
    complete gap-sonnet-terraspec && break
    $NS gap --arms terra-specialist --model claude-sonnet-5-5 --name gap-sonnet-terraspec --threads 3 >> tmp/runs/noharm_system_gap-terraspec.log 2>&1
    complete gap-sonnet-terraspec && break; sleep 300
  done
  $NS concat --arms terra-specialist,gap-sonnet-terraspec --name concat2-terraspec-gap
  $P scripts/noharm_kit.py score --name concat2-terraspec-gap --judge claude-sonnet-5-5 --threads 6 > tmp/runs/noharm_score_concat2-terraspec-gap.log 2>&1
  .venv/bin/python scripts/noharm_kit.py compare --a terra-specialist --b concat2-terraspec-gap > tmp/runs/noharm_compare_concat2-terraspec-gap.json 2>&1
  echo "T7 concat2-terraspec-gap done $(date -u +%H:%M)" ) &
# Codex: grid (T3), then components on terra (T4), the cap (T5) and a regeneration (T6).
( persona astra-specialist gpt-6-astra gpt-6-astra specialist
  persona sol-specialist gpt-6.1-sol gpt-6.1-sol specialist
  persona terra-specialist-r2 gpt-5.6-terra gpt-5.6-terra specialist ) &
( persona terra-format gpt-5.6-terra gpt-5.6-terra format
  persona terra-framing gpt-5.6-terra gpt-5.6-terra framing
  persona terra-length gpt-5.6-terra gpt-5.6-terra length
  persona terra-specialist500 gpt-5.6-terra gpt-5.6-terra specialist500 ) &
# Free models, public cases only (T3; Muse is also the uncapped half of the T5 contrast).
( persona muse-specialist muse-spark-1.3 opencode/muse-spark-1.3-contributor-free specialist
  persona nemotron-specialist nemotron-3-ultra opencode/nemotron-3-ultra-free specialist
  persona pickle-specialist big-pickle opencode/big-pickle specialist ) &
wait
echo "PAPER-A CORE DONE $(date -u +%H:%M)"
# Deferred: second regeneration and repeated sampling (T8), exhaustive headings, Sonnet cap.
( persona terra-specialist-r3 gpt-5.6-terra gpt-5.6-terra specialist
  $NS concat --arms terra-specialist,terra-specialist-r2,terra-specialist-r3 --name selfmoa-terra3
  $P scripts/noharm_kit.py score --name selfmoa-terra3 --judge claude-sonnet-5-5 --threads 6 > tmp/runs/noharm_score_selfmoa-terra3.log 2>&1
  echo "T8 selfmoa-terra3 done $(date -u +%H:%M)"
  persona terra-exhaustive gpt-5.6-terra gpt-5.6-terra exhaustive ) &
( persona sonnet-exhaustive claude-sonnet-5-5 claude-sonnet-5-5 exhaustive
  persona sonnet-specialist500 claude-sonnet-5-5 claude-sonnet-5-5 specialist500 ) &
wait
echo "PAPER-A ALL DONE $(date -u +%H:%M)"
