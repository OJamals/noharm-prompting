# Asking for the Whole Plan — data and code

Materials for *Asking for the Whole Plan: Prompt Design Moves Language Models' Scores on a
Clinical-Safety Benchmark More Than Model Choice. An exploratory study on the NOHARM v2 public
cases*, by Ahmad Omar Aljamal, MD. Preprint: https://www.preprints.org/manuscript/202610.1607.

The main branch carries version 2 of the manuscript. Version 2 repeats the physician review item by
item, and its results replace the overall verdict version 1 reported, that no added option was
harmful; no model output, evaluation or benchmark score changed. Version 1 as posted on October 16,
2026 (doi:10.20944/preprints202610.1607.v1) is the tag `preprint-v1`.

Everything here concerns the 30 public NOHARM v2 cases (330 items). No held-out case was used.
Scores come from a locally run AI evaluator, not from the benchmark organizers, and nothing here is
a clinical validation.

## Contents

| Path | What it is |
|---|---|
| `docs/paper/` | Version 2 (`whole-plan-v2.tex`, its Supplementary Appendix `whole-plan-v2-supplementary-appendix.tex`, and both PDFs); version 1 as posted (`coverage.tex`, `coverage.pdf`); generated numbers, references, figures |
| `sources/mast/results/raw/donoharm/{default,thorough}/` | Every analysed arm: `<arm>.jsonl` holds the model's plans, `<arm>_judged.jsonl` the evaluator's matching and metrics. `-jterra` and `-jfree` arms are the same plans re-scored by GPT-5.6 Terra and Nemotron 3 Ultra |
| `datasets/eval/noharm_offrubric/` | Ratings of off-rubric actions by Nemotron 3 Ultra and Ling 3.1 Flash |
| `datasets/eval/noharm_review_packet/` | Physician review: its unblinding key, the reviewer's item-by-item verdicts (`physician_verdicts.json`, which also records the version 1 verdict it supersedes), the version 1 review page (`review_page.html`), and the clinician packet CSVs |
| `scripts/` | Instructions (`noharm_system.py`), model and judge calls (`noharm_kit.py`), off-rubric audit, and the two offline generators that write every number in the paper |
| `runs/` | The shell scripts that ran the generation and evaluation arms (provenance; they call model endpoints) |
| `site/` | The project page (https://noharm.aljamal.work): `build.py` writes `index.html` from the manuscript's generated numbers |
| `sources/mast/benchmarks/donoharm/` | The NOHARM open-subset kit as used (cases, rubrics, metric code), by the ARISE Network |

Every instruction given to a model is reproduced verbatim in Section S3 of the Supplementary
Appendix (Appendix A of version 1) and defined in `scripts/noharm_system.py`.

## Reproduce the paper's numbers

No model calls and no network access are needed:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m scripts.noharm_paper_snapshot --partial --tex docs/paper/coverage-numbers.tex
.venv/bin/python -m scripts.noharm_paper_analyses --tex docs/paper/coverage-analyses.tex \
    --figures docs/paper/figures --packet datasets/eval/noharm_review_packet
.venv/bin/python docs/paper/jamia/figure_gains.py
cd docs/paper && tectonic whole-plan-v2.tex && tectonic whole-plan-v2-supplementary-appendix.tex
```

Compile the manuscript before its appendix, which reads section names from `whole-plan-v2.aux`.
`tectonic coverage.tex` still compiles version 1 from the same files, with its text unchanged; to
reproduce it exactly, check out `preprint-v1`.

The regenerated `coverage-numbers.tex` and `coverage-analyses.tex` match the committed ones
exactly, except two values. `covFirstJudged` and `covLastJudged`, the dates the paper gives for
when the runs happened, are read from file modification times, which a clone resets. Intervals
resample whole cases with a fixed seed (7), so they reproduce exactly.

Arms the generators list as "pending" were planned but not run or not completed; the paper does
not analyze them. One code path in `scripts/noharm_system.py` (label-grounded correction) imports
a module that is not included. No arm in the paper uses it.

## Re-running models

`runs/*.sh` and `scripts/noharm_kit.py` show how each arm was generated and judged. Models were
reached through the author's own accounts behind an OpenAI-compatible proxy, so re-running needs
your own endpoints; outputs will differ, because the routes expose no fixed checkpoint.

## Licenses

- Code: MIT (`LICENSE`).
- Data produced in this study: CC BY 4.0 (`LICENSE-DATA`).
- NOHARM kit: code MIT and data CC BY 4.0, Copyright (c) 2026 ARISE Network
  (`sources/mast/benchmarks/donoharm/LICENSE`, `LICENSE-DATA`). Cite the NOHARM authors when
  using the cases or rubrics.
