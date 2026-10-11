#!/usr/bin/env bash
# simpledoctor-drky.26.1: the within-model prompting contrasts re-judged by a free model from a
# family no scored arm comes from. usage: noharm_paperA_free.sh <openrouter/model>
cd "$(dirname "$0")/.."
judge=$1
export NOHARM_SLOTS_OPENROUTER=${NOHARM_SLOTS_OPENROUTER:-8}
J="bash tmp/runs/noharm_rejudge.sh"
( $J gpt-5.6-terra gpt-5.6-terra-jfree "$judge"; $J claude-sonnet-5-5 claude-sonnet-5-5-jfree "$judge" ) &
( $J terra-specialist terra-specialist-jfree "$judge"; $J sonnet-specialist sonnet-specialist-jfree "$judge" ) &
wait
echo "FREE JUDGE DONE $(date -u +%H:%M)"
