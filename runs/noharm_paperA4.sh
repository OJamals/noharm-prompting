#!/usr/bin/env bash
# simpledoctor-drky.26.4, trimmed plan (2026-10-11, owner): the only paid test the paper needs is
# the prompt-component split on the two study models, terra and Sonnet 5.5 — 'format' (plan
# structure alone), 'framing' (specialist consult alone), 'length' (>=25 numbered recommendations,
# no clinical structure). Everything else in noharm_paperA3.sh was stopped and is reported as a
# limitation. Generations and judge caches resume where v2/v3 left them. Sonnet judges all six.
cd "$(dirname "$0")/.."
export NOHARM_SLOTS_ANTHROPIC=${NOHARM_SLOTS_ANTHROPIC:-12} NOHARM_SLOTS_CODEX=${NOHARM_SLOTS_CODEX:-5}
S="bash tmp/runs/noharm_system_patient.sh"
persona() { $S "$1" "$2" propose --model "$3" --persona "$4" --threads 3; }
for part in format framing length; do
  persona "sonnet-$part" claude-sonnet-5-5 claude-sonnet-5-5 "$part" &
  persona "terra-$part" gpt-5.6-terra gpt-5.6-terra "$part" &
done
wait
echo "PAPER-A4 DONE $(date -u +%H:%M)"
