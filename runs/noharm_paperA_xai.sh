#!/usr/bin/env bash
# xAI lane for simpledoctor-drky.26, started only once the account answers a full-size prompt:
# a pass against a 503ing account spends ~5 min per item on retries. Probes every 15 min, 48 h max.
cd "$(dirname "$0")/.."
probe() { .venv/bin/python -c "
import sys; sys.path.insert(0,'scripts'); import noharm_system as s
cases=s.load_cases(); item=sorted(cases)[0]
text,_=s.chat('grok-4.7', s.proposal_prompt(cases[item],'specialist'), max_tokens=12000, temperature=0.0)
raise SystemExit(0 if len(text) > 500 else 1)" >/dev/null 2>&1; }
for attempt in $(seq 1 192); do
  probe && break
  echo "XAI down $(date -u +%H:%M)"; sleep 900
done
probe || { echo "XAI never recovered"; exit 1; }
echo "XAI up $(date -u +%H:%M)"
bash tmp/runs/noharm_system_patient.sh grok-specialist grok-4.7 propose --model grok-4.7 --persona specialist --threads 2
bash tmp/runs/noharm_rejudge.sh gpt55-example gpt55-example-grok grok-4.7 All001
bash tmp/runs/noharm_rejudge.sh gpt-5.6-terra gpt-5.6-terra-jgrok grok-4.7
bash tmp/runs/noharm_rejudge.sh terra-specialist terra-specialist-jgrok grok-4.7
echo "XAI LANE DONE $(date -u +%H:%M)"
