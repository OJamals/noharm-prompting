# Asking for the Whole Plan — data and code

Materials for *Asking for the Whole Plan: Prompt Design Moves Language Models' Scores on a
Clinical-Safety Benchmark More Than Model Choice. An exploratory study on the NOHARM v2 public
cases*, by Ahmad Omar Aljamal, MD.

Everything here concerns the 30 public NOHARM v2 cases (330 items). No held-out case was used.
Scores come from a locally run AI evaluator, not from the benchmark organisers, and nothing here is
a clinical validation.

## Contents

| Path | What it is |
|---|---|
| `docs/paper/` | Manuscript source (`coverage.tex`), generated numbers, references, figures |
| `sources/mast/results/raw/donoharm/{default,thorough}/` | Every analysed arm: `<arm>.jsonl` holds the model's plans, `<arm>_judged.jsonl` the evaluator's matching and metrics. `-jterra` and `-jfree` arms are the same plans re-scored by GPT-5.6 terra and Nemotron 3 Ultra |
| `datasets/eval/noharm_offrubric/` | Ratings of off-rubric actions by Nemotron 3 Ultra and Ling 3.1 Flash |
| `datasets/eval/noharm_review_packet/` | Physician review: the review page (`review_page.html`), its unblinding key, the reviewer's verdicts, and the clinician packet CSVs |
| `scripts/` | Instructions (`noharm_system.py`), model and judge calls (`noharm_kit.py`), off-rubric audit, and the two offline generators that write every number in the paper |
| `runs/` | The shell scripts that ran the generation and evaluation arms (provenance; they call model endpoints) |
| `sources/mast/benchmarks/donoharm/` | The NOHARM open-subset kit as used (cases, rubrics, metric code), by the ARISE Network |

Every instruction given to a model is reproduced verbatim in the paper's Appendix A and defined in
`scripts/noharm_system.py`.

## Reproduce the paper's numbers

No model calls and no network access are needed:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m scripts.noharm_paper_snapshot --partial --tex docs/paper/coverage-numbers.tex
.venv/bin/python -m scripts.noharm_paper_analyses --tex docs/paper/coverage-analyses.tex \
    --figures docs/paper/figures --packet datasets/eval/noharm_review_packet
cd docs/paper && tectonic coverage.tex
```

The regenerated `coverage-numbers.tex` and `coverage-analyses.tex` match the committed ones
exactly, except two values. `covFirstJudged` and `covLastJudged`, the dates the paper gives for
when the runs happened, are read from file modification times, which a clone resets. Intervals
resample whole cases with a fixed seed (7), so they reproduce exactly.

Arms the generators list as "pending" were planned but not run or not completed; the paper does
not analyse them. One code path in `scripts/noharm_system.py` (label-grounded correction) imports
a module that is not included. No arm in the paper uses it.

## Re-running models

`runs/*.sh` and `scripts/noharm_kit.py` show how each arm was generated and judged. Models were
reached through the author's own accounts behind an OpenAI-compatible proxy, so re-running needs
your own endpoints; outputs will differ, because the routes expose no fixed checkpoint.

## Licences

- Code: MIT (`LICENSE`).
- Data produced in this study: CC BY 4.0 (`LICENSE-DATA`).
- NOHARM kit: code MIT and data CC BY 4.0, Copyright (c) 2026 ARISE Network
  (`sources/mast/benchmarks/donoharm/LICENSE`, `LICENSE-DATA`). Cite the NOHARM authors when
  using the cases or rubrics.
