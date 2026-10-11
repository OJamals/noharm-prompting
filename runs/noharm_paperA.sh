#!/usr/bin/env bash
# simpledoctor-drky.26: every arm the prompting paper needs, in lanes per proxy account. Each arm is
# generated until complete, Sonnet-judged, and compared with its baseline; account slots in
# noharm_kit.chat keep the lanes from oversubscribing an account. Log: tmp/runs/noharm_paperA.log
cd "$(dirname "$0")/.."
S="bash tmp/runs/noharm_system_patient.sh"
K="bash tmp/runs/noharm_kit_patient.sh"
J="bash tmp/runs/noharm_rejudge.sh"
NS=".venv/bin/python scripts/noharm_system.py"
P=sources/mast/benchmarks/donoharm/.venv/bin/python
D=sources/mast/results/raw/donoharm/default
persona() {  # <name> <baseline> <model> <persona>
  $S "$1" "$2" propose --model "$3" --persona "$4" --threads 2
}
complete() { [ "$(grep -c . "$D/$1.jsonl" 2>/dev/null || echo 0)" -ge 330 ]; }
scored() { grep -q "Overall *F1_weighted" "tmp/runs/noharm_score_$1.log" 2>/dev/null; }

# --- Anthropic: T3 grid, T4 factorial (Sonnet), T5 caps, T6 re-judge, T7 completion ----------
( persona sonnet-specialist claude-sonnet-5-5 claude-sonnet-5-5 specialist
  persona sonnet-format claude-sonnet-5-5 claude-sonnet-5-5 format
  persona sonnet-framing claude-sonnet-5-5 claude-sonnet-5-5 framing
  persona sonnet-length claude-sonnet-5-5 claude-sonnet-5-5 length
  persona sonnet-exhaustive claude-sonnet-5-5 claude-sonnet-5-5 exhaustive ) &
( persona opus-specialist claude-opus-5-5 claude-opus-5-5 specialist
  persona haiku-specialist claude-haiku-4-5-20251001 claude-haiku-4-5-20251001 specialist
  persona sonnet5-specialist claude-sonnet-5 claude-sonnet-5 specialist
  persona sonnet-specialist500 claude-sonnet-5-5 claude-sonnet-5-5 specialist500
  $K sonnet-thorough claude-sonnet-5-5 thorough claude-sonnet-5-5 ) &
( $J terra-specialist terra-specialist-rj claude-sonnet-5-5
  for pass in $(seq 1 20); do
    complete gap-sonnet-terraspec && break
    $NS gap --arms terra-specialist --model claude-sonnet-5-5 --name gap-sonnet-terraspec --threads 2 >> tmp/runs/noharm_system_gap-terraspec.log 2>&1
    complete gap-sonnet-terraspec && break; sleep 300
  done
  $NS concat --arms terra-specialist,gap-sonnet-terraspec --name concat2-terraspec-gap
  $P scripts/noharm_kit.py score --name concat2-terraspec-gap --judge claude-sonnet-5-5 --threads 6 > tmp/runs/noharm_score_concat2-terraspec-gap.log 2>&1
  .venv/bin/python scripts/noharm_kit.py compare --a terra-specialist --b concat2-terraspec-gap > tmp/runs/noharm_compare_concat2-terraspec-gap.json 2>&1
  echo "T7 concat2-terraspec-gap done $(date -u +%H:%M)" ) &

# --- Codex: T3 grid, T4 factorial (terra), T5 cap, T6 regenerations, T1 terra judge -----------
( persona luna-specialist gpt-5.6-luna gpt-5.6-luna specialist
  persona terra-format gpt-5.6-terra gpt-5.6-terra format
  persona terra-framing gpt-5.6-terra gpt-5.6-terra framing
  persona terra-length gpt-5.6-terra gpt-5.6-terra length
  persona terra-exhaustive gpt-5.6-terra gpt-5.6-terra exhaustive
  persona terra-specialist500 gpt-5.6-terra gpt-5.6-terra specialist500 ) &
( persona astra-specialist gpt-6-astra gpt-6-astra specialist
  persona sol-specialist gpt-6.1-sol gpt-6.1-sol specialist
  persona terra-specialist-r2 gpt-5.6-terra gpt-5.6-terra specialist
  persona terra-specialist-r3 gpt-5.6-terra gpt-5.6-terra specialist
  $NS concat --arms terra-specialist,terra-specialist-r2,terra-specialist-r3 --name selfmoa-terra3
  $P scripts/noharm_kit.py score --name selfmoa-terra3 --judge claude-sonnet-5-5 --threads 6 > tmp/runs/noharm_score_selfmoa-terra3.log 2>&1
  .venv/bin/python scripts/noharm_kit.py compare --a concat3-prompted --b selfmoa-terra3 > tmp/runs/noharm_compare_selfmoa-terra3.json 2>&1
  echo "T8 selfmoa-terra3 done $(date -u +%H:%M)"
  $K gpt-5.6-terra-r2 gpt-5.6-terra default gpt-5.6-terra
  $J gpt-5.6-terra gpt-5.6-terra-jterra gpt-5.6-terra
  $J sonnet-pcp sonnet-pcp-jterra gpt-5.6-terra ) &

# --- xAI: T3 grid, T1 third-family judge (calibrated on the kit's Gemini-judged example first) --
( persona grok-specialist grok-4.7 grok-4.7 specialist
  $J gpt55-example gpt55-example-grok grok-4.7 All001
  $J gpt-5.6-terra gpt-5.6-terra-jgrok grok-4.7
  $J terra-specialist terra-specialist-jgrok grok-4.7 ) &

# --- OpenCode free models (public data, development only): T3 grid, T5 Muse uncapped -----------
( persona muse-specialist muse-spark-1.3 opencode/muse-spark-1.3-contributor-free specialist
  persona nemotron-specialist nemotron-3-ultra opencode/nemotron-3-ultra-free specialist
  persona pickle-specialist big-pickle opencode/big-pickle specialist ) &

wait
echo "PAPER-A ALL DONE $(date -u +%H:%M)"
