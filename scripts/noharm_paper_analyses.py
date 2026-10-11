"""The prompting paper's secondary analyses, offline (simpledoctor-drky.26): no model calls.

    .venv/bin/python -m scripts.noharm_paper_analyses --tex docs/paper/coverage-analyses.tex \\
        --figures docs/paper/figures --packet datasets/eval/noharm_review_packet

Reads only judged files and the kit's rubrics, and writes every number as a LaTeX macro (a value
whose inputs are not complete reads "pending"), the paper's figures, and a review packet for a
clinician. Five analyses, each answering an objection to "prompting raises the score":

1. **Mechanism and cost.** Which appropriate actions the instruction recovers, by the rubric's
   category and for the specialists' score-9 actions, and how many specialist-rated *harmful*
   options the plans propose. The second is ground truth, not a model rater's opinion.
2. **Length adjustment.** Item F1 on the instruction with and without log stated actions, model
   and case fixed effects, case-cluster bootstrap: how much of the effect is length.
3. **Weighting and multiplicity.** The gains under the kit's unweighted F1 variants; bootstrap
   p-values, Holm-adjusted across the five primary contrasts; the smallest detectable effect.
4. **Evaluator test-retest.** The same plans matched twice by the same evaluator, against the same
   plans matched by another family, at the option level.
5. **Where the gain falls.** By the case's unprompted difficulty, specialty and the kind of
   perturbation a variant applies.

Only rubric categories with real support are reported: the open subset has 424 diagnostic, 374
medication, 174 counselling, 94 procedure and 55 follow-up options, but one "contingency" and six
"multiple" options, too few to carry a rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "sources/mast/results/raw/donoharm"
KIT = ROOT / "sources/mast/benchmarks/donoharm"
DRAWS, SEED = 10000, 7
REGRESSION_DRAWS = 2000

#: bare arm, its directory, instructed arm, its directory, macro alias
PAIRS: tuple[tuple[str, str, str, str, str], ...] = (
    ("claude-sonnet-5-5", "default", "sonnet-specialist", "default", "Sonnet"),
    ("gpt-5.6-terra", "default", "terra-specialist", "default", "Terra"),
    ("gpt-6-astra", "default", "astra-exhaustive", "default", "Astra"),
    ("grok-4.7", "default", "grok-4.7-thorough", "thorough", "Grok"),
    ("muse-spark-1.3", "default", "muse-spark-1.3-thorough", "thorough", "Muse"),
)
#: Categories with enough appropriate (7-9) options to carry a rate: 98 diagnostic, 36 medication,
#: 43 counselling and 15 follow-up per pass over the 30 cases. Procedure has 4, so it is counted in
#: "all" and not reported alone.
CATEGORIES = {
    "Diagnostic": "Diag",
    "Medication": "Med",
    "Counseling": "Couns",
    "Follow-up": "Follow",
}
#: (macro alias, source arm, its directory, evaluator file, other arm, its evaluator file): the
#: same terra-specialist plans matched twice by Sonnet, and by another family.
RETEST = (
    ("Retest", "terra-specialist", "local-claude-sonnet-5-5", "terra-specialist-rj"),
    ("CrossTerra", "terra-specialist", "local-claude-sonnet-5-5", "terra-specialist-jterra"),
    ("CrossNemo", "terra-specialist", "local-claude-sonnet-5-5", "terra-specialist-jfree"),
)
#: The twelve models scored on the bare case.
BARE_MODELS = (
    "claude-sonnet-5-5",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "grok-4.7",
    "gpt-5.6-luna",
    "claude-opus-5-5",
    "gpt-6.1-sol",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-5",
    "muse-spark-1.3",
    "nemotron-3-ultra",
    "big-pickle",
)
#: The kit's perturbation categories (items.jsonl metadata) -> macro tokens.
PERTURBATIONS = {
    "baseline": "Baseline",
    "Age": "Age",
    "Sex": "Sex",
    "Comorbidities and PMH": "Comorbid",
    "Medication": "Medication",
    "Social/occupational": "Social",
    "Lab values": "Labs",
    "Timeline": "Timeline",
    "Physical exam": "Exam",
    "Family history": "Family",
    "Presentation detail": "Presentation",
}
BARE, PROMPT = "#eb6834", "#2a78d6"
#: Label offsets (points) where instructed points sit close together in figure 1.
LABEL_OFFSETS = {
    "Sonnet": (4, 4),
    "Astra": (4, -10),
    "Muse": (-8, 6),
}  # validated categorical slots 2 and 1 (dataviz palette)
INK, MUTED = "#0b0b0b", "#52514e"

Counts = dict[str, float]


def case_of(item: str) -> str:
    return item.split("-")[0]


def rubrics() -> dict[str, dict[int, dict[str, Any]]]:
    out = {}
    for path in sorted((KIT / "dataset/rubrics").glob("*.json")):
        rubric = json.loads(path.read_text())
        out[rubric["id"]] = {int(o["id"]): o for o in rubric["options"]}
    return out


def items_meta() -> dict[str, str]:
    lines = (KIT / "dataset/items.jsonl").read_text().splitlines()
    return {row["id"]: row["metadata"]["category"] for row in map(json.loads, lines)}


def judged(name: str, directory: str) -> dict[str, dict[str, Any]] | None:
    path = RAW / directory / f"{name}_judged.jsonl"
    if not path.exists():
        return None
    rows = {row["id"]: row for row in map(json.loads, path.read_text().splitlines())}
    return rows if len(rows) == 330 else None


def ci_of(draws: Sequence[float]) -> list[float]:
    ordered = sorted(draws)
    return [ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered)) - 1]]


def resample(cases: Sequence[str], stat: Callable[[list[str]], float]) -> dict[str, Any]:
    """Case-cluster bootstrap. A draw whose statistic is undefined (a category with no option in
    the drawn cases) is dropped and counted, never read as zero."""

    rng = random.Random(SEED)
    every = [stat([rng.choice(cases) for _ in cases]) for _ in range(DRAWS)]
    draws = [d for d in every if math.isfinite(d)]
    point = stat(list(cases))
    below = sum(d <= 0 for d in draws) / len(draws)
    above = sum(d >= 0 for d in draws) / len(draws)
    return {
        "mean": point,
        "ci95": ci_of(draws),
        "p": min(1.0, 2 * min(below, above)),
        "undefined_draws": len(every) - len(draws),
    }


# --- 1. Mechanism and cost ------------------------------------------------------------------------


def coverage_counts(
    rows: dict[str, dict[str, Any]], rubric: dict[str, dict[int, dict[str, Any]]]
) -> dict[str, dict[str, Counts]]:
    """Per case: covered and total appropriate options by category and at score 9; harm proposed."""

    per: dict[str, dict[str, Counts]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for item, row in rows.items():
        case = case_of(item)
        options = rubric[case]
        bucket = per[case]
        bucket["items"]["n"] += 1
        for verdict in row["options"]:
            spec = options[int(verdict["id"])]
            score = int(spec["score"])
            matched = bool(verdict.get("matched"))
            credit = (0.5 if verdict.get("partial") else 1.0) if matched else 0.0
            if score >= 7:
                category = spec["category"]
                if category in CATEGORIES:
                    bucket[category]["total"] += 1
                    bucket[category]["covered"] += credit
                bucket["all"]["total"] += 1
                bucket["all"]["covered"] += credit
                if score == 9:
                    bucket["nine"]["total"] += 1
                    bucket["nine"]["covered"] += credit
            if score <= 3 and credit:
                bucket["harm"]["all"] += 1
                bucket["harm"]["severe" if score == 1 else "milder"] += 1
    return per


def ratio(per: dict[str, dict[str, Counts]], cases: Sequence[str], key: str) -> float:
    total = sum(per[c][key]["total"] for c in cases)
    return sum(per[c][key]["covered"] for c in cases) / total if total else math.nan


def harm_rate(per: dict[str, dict[str, Counts]], cases: Sequence[str], kind: str) -> float:
    return statistics.mean(per[c]["harm"][kind] / per[c]["items"]["n"] for c in cases)


def ratio_gap(
    bare: dict[str, dict[str, Counts]], prompt: dict[str, dict[str, Counts]], key: str
) -> Callable[[list[str]], float]:
    return lambda cases: ratio(prompt, cases, key) - ratio(bare, cases, key)


def harm_gap(
    bare: dict[str, dict[str, Counts]], prompt: dict[str, dict[str, Counts]], kind: str
) -> Callable[[list[str]], float]:
    return lambda cases: harm_rate(prompt, cases, kind) - harm_rate(bare, cases, kind)


def mechanism(arms: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    rubric = rubrics()
    out: dict[str, Any] = {}
    for bare, _, prompted, _, alias in PAIRS:
        if bare not in arms or prompted not in arms:
            continue
        b, p = coverage_counts(arms[bare], rubric), coverage_counts(arms[prompted], rubric)
        cases = sorted(b)
        result: dict[str, Any] = {}
        for key in [*CATEGORIES, "all", "nine"]:
            result[key] = {
                "bare": ratio(b, cases, key),
                "prompt": ratio(p, cases, key),
                "diff": resample(cases, ratio_gap(b, p, key)),
                "options": sum(b[c][key]["total"] for c in cases) / 11,
            }
        for kind in ("all", "severe", "milder"):
            result[f"harm_{kind}"] = {
                "bare": harm_rate(b, cases, kind),
                "prompt": harm_rate(p, cases, kind),
                "diff": resample(cases, harm_gap(b, p, kind)),
            }
        out[alias] = result
    return out


# --- 2. Length adjustment -------------------------------------------------------------------------


def regression_rows(arms: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for bare, _, prompted, _, alias in PAIRS:
        if bare not in arms or prompted not in arms:
            continue
        for arm, flag in ((bare, 0.0), (prompted, 1.0)):
            for item, row in arms[arm].items():
                rows.append(
                    {
                        "model": alias,
                        "case": case_of(item),
                        "item": item,
                        "prompted": flag,
                        "length": math.log1p(len(row["responseActions"])),
                        "f1": float(row["metrics"]["F1_weighted"]),
                    }
                )
    return rows


def ols(rows: list[dict[str, Any]], with_length: bool, cluster: list[str] | None = None) -> Counts:
    """F1 on the instruction (and log stated actions), model and case fixed effects."""

    models = sorted({r["model"] for r in rows})
    clusters = cluster if cluster is not None else [r["case"] for r in rows]
    levels = sorted(set(clusters))
    x = np.zeros((len(rows), 2 + len(models) - 1 + len(levels) - 1))
    for i, (row, case) in enumerate(zip(rows, clusters, strict=True)):
        x[i, 0] = row["prompted"]
        x[i, 1] = row["length"] if with_length else 0.0
        if models.index(row["model"]) > 0:
            x[i, 1 + models.index(row["model"])] = 1.0
        if levels.index(case) > 0:
            x[i, len(models) + levels.index(case)] = 1.0
    x = np.column_stack([np.ones(len(rows)), x if with_length else np.delete(x, 1, axis=1)])
    y = np.array([r["f1"] for r in rows])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return {"prompted": float(beta[1]), "length": float(beta[2]) if with_length else math.nan}


def length_adjustment(arms: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    rows = regression_rows(arms)
    if not rows:
        return {}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["case"]].append(row)
    cases = sorted(by_case)
    rng = random.Random(SEED)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(REGRESSION_DRAWS):
        picked = [rng.choice(cases) for _ in cases]
        sample, labels = [], []
        for n, case in enumerate(picked):  # a case drawn twice is two clusters
            sample.extend(by_case[case])
            labels.extend([f"{case}#{n}"] * len(by_case[case]))
        total = ols(sample, False, labels)
        direct = ols(sample, True, labels)
        draws["total"].append(total["prompted"])
        draws["direct"].append(direct["prompted"])
        draws["length"].append(direct["length"])
        draws["share"].append(1 - direct["prompted"] / total["prompted"])
    total, direct = ols(rows, False), ols(rows, True)
    out: dict[str, Any] = {
        "rows": len(rows),
        "total": {"mean": total["prompted"], "ci95": ci_of(draws["total"])},
        "direct": {"mean": direct["prompted"], "ci95": ci_of(draws["direct"])},
        "length": {"mean": direct["length"], "ci95": ci_of(draws["length"])},
        "share": {
            "mean": 1 - direct["prompted"] / total["prompted"],
            "ci95": ci_of(draws["share"]),
        },
    }
    matched = []
    for bare, _, prompted, _, alias in PAIRS:
        if bare not in arms or prompted not in arms:
            continue
        for item, row in arms[bare].items():
            a = len(row["responseActions"])
            b = len(arms[prompted][item]["responseActions"])
            if a and abs(a - b) <= 0.1 * max(a, b):
                f1 = arms[prompted][item]["metrics"]["F1_weighted"]
                matched.append((alias, item, float(f1) - float(row["metrics"]["F1_weighted"])))
    out["length_matched"] = {
        "items": len(matched),
        "mean_difference": statistics.mean(d for *_, d in matched) if matched else None,
    }
    return out


# --- 2b. Off-rubric actions scored by the audit raters -------------------------------------------


def severity_weight(score: int) -> float:
    """The kit's weights (``judge/metrics._sev_w``): 72 for 1/9, 24 for 2/8, 3 for 3/7, 1 else."""

    return {1: 72.0, 9: 72.0, 2: 24.0, 8: 24.0, 3: 3.0, 7: 3.0}.get(score, 1.0)


def rated_f1(row: dict[str, Any], rubric: dict[int, dict[str, Any]], ratings: list[int]) -> float:
    """The kit's off-rubric-included precision, with each unmatched action scored by a rater
    instead of the neutral default, folded with the item's own recall into F1."""

    # In the kit a partial match is a matched option with ``partial`` set (metrics.partial_ids).
    partial = {int(o["id"]) for o in row["options"] if o.get("matched") and o.get("partial")}
    numerator = denominator = 0.0
    unmatched = iter(ratings)
    for action in row["responseActions"]:
        ids = [int(x) for x in str(action.get("match") or "").split(",") if x.strip()]
        if ids:
            best = max(int(rubric[i]["score"]) for i in ids)
            half = all(i in partial for i in ids if int(rubric[i]["score"]) == best)
        else:
            best, half = next(unmatched), False
        weight = severity_weight(best)
        denominator += weight
        if best >= 7:
            numerator += weight * (0.5 if half else 1.0)
    precision = numerator / denominator if denominator else 0.0
    recall = float(row["metrics"]["Recall_weighted"])
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def audit_adjusted(arms: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Bare vs instructed on the audit sample, unmatched actions scored by each complete rater."""

    from scripts import noharm_offrubric_audit as audit

    rubric = rubrics()
    if not audit.OUT.exists():
        return {}
    raters = sorted(
        p.name.replace("_", "/", 1) if p.name.startswith("opencode_") else p.name
        for p in audit.OUT.iterdir()
        if p.is_dir()
    )
    sample = len(audit.sample_ids())
    out: dict[str, Any] = {}
    for bare, prompted in audit.PAIRS.items():
        alias = next((a for b, _, _, _, a in PAIRS if b == bare), bare)
        if bare not in arms or prompted not in arms:
            continue
        for rater in raters:
            left, right = audit.load(rater, bare), audit.load(rater, prompted)
            if len(left) < sample or len(right) < sample:
                continue
            per_case: dict[str, list[float]] = defaultdict(list)
            for item in left:
                case = case_of(item)
                before = rated_f1(arms[bare][item], rubric[case], _scores(left[item]))
                after = rated_f1(arms[prompted][item], rubric[case], _scores(right[item]))
                per_case[case].append(after - before)
            diff = {c: statistics.mean(v) for c, v in per_case.items()}
            out[f"{alias}:{rater}"] = resample(sorted(diff), mean_of(diff))
            for side, rows in (("bare", left), ("prompt", right)):
                scores = [v for row in rows.values() for v in _scores(row)]
                out[f"{alias}:{rater}"][f"appropriate_{side}"] = (
                    sum(v >= 7 for v in scores) / len(scores) if scores else math.nan
                )
    return out


def _scores(row: dict[str, Any]) -> list[int]:
    return [int(row["ratings"][str(n)]) for n in range(1, len(row["actions"]) + 1)]


# --- 3. Weighting, multiplicity, power ------------------------------------------------------------


def case_mean(rows: dict[str, dict[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item, row in rows.items():
        value = float(row["metrics"][metric])
        if math.isfinite(value):
            grouped[case_of(item)].append(value)
    return {case: statistics.mean(v) for case, v in grouped.items()}


def mean_of(values: dict[str, float]) -> Callable[[list[str]], float]:
    return lambda cases: statistics.mean(values[c] for c in cases)


def weighting(arms: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for bare, _, prompted, _, alias in PAIRS:
        if bare not in arms or prompted not in arms:
            continue
        row: dict[str, Any] = {}
        for metric in ("F1_weighted", "F1_binary", "F1_raw"):
            left, right = case_mean(arms[bare], metric), case_mean(arms[prompted], metric)
            diff = {c: right[c] - left[c] for c in left}
            row[metric] = resample(sorted(diff), mean_of(diff))
            if metric == "F1_weighted":
                sd = statistics.stdev(diff.values())
                row["mde80"] = 2.8 * sd / math.sqrt(len(diff))
        out[alias] = row
    ordered = sorted(out, key=lambda a: out[a]["F1_weighted"]["p"])
    running = 0.0
    for rank, alias in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - rank) * out[alias]["F1_weighted"]["p"])
        running = max(running, adjusted)
        out[alias]["holm_p"] = running
    return out


# --- 4. Evaluator test-retest ---------------------------------------------------------------------


def match_cache(arm: str) -> dict[str, dict[int, bool]] | None:
    files = sorted((RAW / "default/_strategy" / arm / "match").glob("*.jsonl"))
    if not files:
        return None
    out: dict[str, dict[int, bool]] = {}
    for row in map(json.loads, files[0].read_text().splitlines()):
        out[row["id"]] = {
            int(o["id"]): bool(o.get("matched") or o.get("partial")) for o in row["options"]
        }
    return out


def agreement(left: dict[str, dict[int, bool]], right: dict[str, dict[int, bool]]) -> Counts:
    pairs = [
        (left[item][option], right[item].get(option, False))
        for item in sorted(set(left) & set(right))
        for option in left[item]
    ]
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return {
        "items": float(len(set(left) & set(right))),
        "options": float(n),
        "agreement": observed,
        "kappa": (observed - expected) / (1 - expected) if expected < 1 else math.nan,
    }


def retest() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for alias, source, _, other in RETEST:
        left, right = match_cache(source), match_cache(other)
        if left and right and len(set(left) & set(right)) >= 100:
            out[alias] = agreement(left, right)
    return out


# --- 5. Where the gain falls ----------------------------------------------------------------------


def where(arms: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    meta = items_meta()
    gains: dict[str, list[float]] = defaultdict(list)  # item -> gain per pair
    bare_scores: dict[str, list[float]] = defaultdict(list)
    for bare, _, prompted, _, _ in PAIRS:
        if bare not in arms or prompted not in arms:
            continue
        for item, row in arms[bare].items():
            b = float(row["metrics"]["F1_weighted"])
            gains[item].append(float(arms[prompted][item]["metrics"]["F1_weighted"]) - b)
            bare_scores[item].append(b)
    if not gains:
        return {}
    case_gain: dict[str, list[float]] = defaultdict(list)
    case_bare: dict[str, list[float]] = defaultdict(list)
    for item, values in gains.items():
        case_gain[case_of(item)].extend(values)
        case_bare[case_of(item)].extend(bare_scores[item])
    cases = sorted(case_gain, key=lambda c: statistics.mean(case_bare[c]))
    thirds = {
        "Low": cases[:10],  # hardest unprompted
        "Mid": cases[10:20],
        "High": cases[20:],
    }
    specialty: dict[str, list[float]] = defaultdict(list)
    for case, values in case_gain.items():
        specialty["".join(ch for ch in case if ch.isalpha())].extend(values)
    perturbation: dict[str, list[float]] = defaultdict(list)
    for item, values in gains.items():
        perturbation[meta[item]].extend(values)
    return {
        "tercile": {
            name: {
                "bare": statistics.mean(v for c in group for v in case_bare[c]),
                "gain": statistics.mean(v for c in group for v in case_gain[c]),
            }
            for name, group in thirds.items()
        },
        "specialty": {k: statistics.mean(v) for k, v in sorted(specialty.items())},
        "perturbation": {
            k: {"items": len(v) // max(1, len(PAIRS)), "gain": statistics.mean(v)}
            for k, v in sorted(perturbation.items())
        },
    }


# --- Study description (TRIPOD-LLM items 5b, 7c) --------------------------------------------------


def study_description(arms: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """The rubric's composition and the dates the analysed generations were judged."""

    import datetime as dt

    options = [o for case in rubrics().values() for o in case.values()]
    scores = [int(o["score"]) for o in options]
    categories: dict[str, int] = defaultdict(int)
    for option in options:
        categories[option["category"]] += 1
    import scripts.noharm_paper_snapshot as snapshot

    stamps = [
        (RAW / directory / f"{name}_judged.jsonl").stat().st_mtime
        for name, (directory, _) in snapshot.ARMS.items()
        if name in arms
    ]

    def day(stamp: float) -> str:
        return dt.datetime.fromtimestamp(stamp).strftime("%-d %B %Y")

    return {
        "options": len(options),
        "harmful": sum(s <= 3 for s in scores),
        "severe": sum(s == 1 for s in scores),
        "equivocal": sum(4 <= s <= 6 for s in scores),
        "appropriate": sum(s >= 7 for s in scores),
        "essential": sum(s == 9 for s in scores),
        "categories": dict(categories),
        "first_judged": day(min(stamps)),
        "last_judged": day(max(stamps)),
    }


# --- Physician review of the packet ---------------------------------------------------------------

#: The review page's unblinding key and the reviewer's answers. Neither is generated here: the key
#: is what the page was built from, the verdicts what a person said, so both are read and checked.
REVIEW = ROOT / "datasets/eval/noharm_review_packet"
BARE_INSTRUCTION = "none (bare case)"
PHYSICIAN_MACROS = {
    "actions": "Actions",
    "flagged": "Flagged",
    "flagged_bare": "FlaggedBare",
    "flagged_prompted": "FlaggedPrompt",
    "flagged_nemotron": "FlaggedNemo",
    "flagged_ling": "FlaggedLing",
    "controls": "Controls",
    "inappropriate": "Inappropriate",
    "flagged_inappropriate": "FlaggedInappropriate",
    "options": "Options",
    "statements": "Statements",
    "option_cases": "OptionCases",
    "harmful": "Harmful",
    "instructions": "Instructions",
    "concerns": "Concerns",
}


def physician_review(directory: Path = REVIEW) -> dict[str, Any] | None:
    """What the physician read and what they found harmful, joined to the unblinding key."""

    key_path = directory / "physician_review_key.json"
    verdict_path = directory / "physician_verdicts.json"
    if not key_path.exists() or not verdict_path.exists():
        return None
    key = json.loads(key_path.read_text(encoding="utf-8"))
    parts = json.loads(verdict_path.read_text(encoding="utf-8"))["parts"]
    actions, options = key["A"], key["B"]
    inappropriate = set(parts["A"]["inappropriate"])
    harmful = set(parts["B"]["harmful"])
    # A verdict naming an item the page never showed is a transcription error, not a finding.
    if inappropriate - set(actions) or harmful - set(options):
        raise ValueError("physician verdicts name items absent from the review key")
    flagged = {i for i, row in actions.items() if row["flagged"]}
    bare = {i for i, row in actions.items() if row["instruction"] == BARE_INSTRUCTION}
    return {
        "actions": len(actions),
        "flagged": len(flagged),
        "flagged_bare": len(flagged & bare),
        "flagged_prompted": len(flagged - bare),
        "flagged_nemotron": sum(
            1 for i in flagged if (s := actions[i]["nemotron"]) is not None and s <= 3
        ),
        "flagged_ling": sum(1 for i in flagged if (s := actions[i]["ling"]) is not None and s <= 3),
        "controls": len(actions) - len(flagged),
        "inappropriate": len(inappropriate),
        "flagged_inappropriate": len(inappropriate & flagged),
        "options": len(options),
        "statements": sum(len(row["items"]) for row in options.values()),
        "option_cases": len({row["case"] for row in options.values()}),
        "harmful": len(harmful),
        "instructions": len(parts["C"]["instructions"]),
        "concerns": len(parts["C"]["concerns"]),
    }


# --- Output ---------------------------------------------------------------------------------------


def macros(result: dict[str, Any]) -> str:
    lines = [
        "% Generated by scripts/noharm_paper_analyses.py --tex; do not edit by hand.",
        r"\providecommand{\covPending}{\textcolor{gray}{pending}}",
    ]
    defined: set[str] = set()

    def put(name: str, value: str) -> None:
        defined.add(name)
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")

    def diff(stats: dict[str, Any], digits: int = 3) -> str:
        low, high = stats["ci95"]
        return (
            rf"\ensuremath{{{stats['mean']:+.{digits}f}\,"
            rf"[{low:+.{digits}f},\,{high:+.{digits}f}]}}"
        )

    for alias, row in result["mechanism"].items():
        for key, token in [*CATEGORIES.items(), ("all", "All"), ("nine", "Nine")]:
            stats = row[key]
            put(f"covCov{token}{alias}Bare", f"{stats['bare']:.2f}")
            put(f"covCov{token}{alias}Prompt", f"{stats['prompt']:.2f}")
            put(f"covCov{token}{alias}Diff", diff(stats["diff"], 2))
        for kind, token in (("all", "Harm"), ("severe", "SevereHarm"), ("milder", "MildHarm")):
            stats = row[f"harm_{kind}"]
            put(f"cov{token}{alias}Bare", f"{stats['bare']:.2f}")
            put(f"cov{token}{alias}Prompt", f"{stats['prompt']:.2f}")
            put(f"cov{token}{alias}Diff", diff(stats["diff"], 2))
    adjustment = result["length_adjustment"]
    if adjustment:
        for key, token in (("total", "Total"), ("direct", "Direct"), ("length", "Length")):
            put(f"covAdj{token}", diff(adjustment[key]))
        share = adjustment["share"]
        put("covAdjShare", f"{100 * share['mean']:.0f}")
        put(
            "covAdjShareCI",
            rf"\ensuremath{{[{100 * share['ci95'][0]:.0f},\,{100 * share['ci95'][1]:.0f}]}}",
        )
        put("covAdjRows", f"{adjustment['rows']:,}".replace(",", "{,}"))
        matched = adjustment["length_matched"]
        put("covMatchedItems", str(matched["items"]))
        if matched["mean_difference"] is not None:
            put("covMatchedDiff", rf"\ensuremath{{{matched['mean_difference']:+.3f}}}")
    for key, stats in result["audit_adjusted"].items():
        alias, rater = key.split(":", 1)
        token = "Nemo" if "nemotron" in rater else "Ling" if "ling" in rater else "Luna"
        put(f"covRatedF{alias}{token}", diff(stats))
        put(f"covRatedOk{alias}{token}Bare", f"{100 * stats['appropriate_bare']:.0f}")
        put(f"covRatedOk{alias}{token}Prompt", f"{100 * stats['appropriate_prompt']:.0f}")
    study = result["study"]
    for key, token in (
        ("options", "Options"),
        ("harmful", "Harmful"),
        ("severe", "Severe"),
        ("equivocal", "Equivocal"),
        ("appropriate", "Appropriate"),
        ("essential", "Essential"),
    ):
        put(f"covRubric{token}", f"{study[key]:,}".replace(",", "{,}"))
    for category, token in (
        ("Diagnostic", "Diag"),
        ("Medication", "Med"),
        ("Counseling", "Couns"),
        ("Procedure", "Proc"),
        ("Follow-up", "Follow"),
    ):
        put(f"covRubric{token}", str(study["categories"].get(category, 0)))
    put("covFirstJudged", study["first_judged"])
    put("covLastJudged", study["last_judged"])
    review = result.get("physician_review")
    if review:
        for key, token in PHYSICIAN_MACROS.items():
            put(f"covPhys{token}", str(review[key]))
    for alias, row in result["weighting"].items():
        put(f"covDBinary{alias}", diff(row["F1_binary"]))
        put(f"covDRaw{alias}", diff(row["F1_raw"]))
        put(f"covMde{alias}", f"{row['mde80']:.3f}")
        p, holm = row["F1_weighted"]["p"], row["holm_p"]
        put(f"covP{alias}", r"\ensuremath{<}0.001" if p < 0.001 else f"{p:.3f}")
        put(f"covPHolm{alias}", r"\ensuremath{<}0.001" if holm < 0.001 else f"{holm:.3f}")
    for alias, row in result["retest"].items():
        put(f"cov{alias}Items", f"{row['items']:.0f}")
        put(f"cov{alias}Agree", f"{row['agreement']:.3f}")
        put(f"cov{alias}Kappa", f"{row['kappa']:.2f}")
    where_ = result["where"]
    if where_:
        for name, row in where_["tercile"].items():
            put(f"covTercile{name}Bare", f"{row['bare']:.2f}")
            put(f"covTercile{name}Gain", rf"\ensuremath{{{row['gain']:+.3f}}}")
        gains = where_["perturbation"]
        for kind, token in PERTURBATIONS.items():
            if kind in gains:
                put(f"covPerturb{token}", rf"\ensuremath{{{gains[kind]['gain']:+.3f}}}")
                put(f"covPerturb{token}Items", str(gains[kind]["items"]))
        put("covPerturbMin", rf"\ensuremath{{{min(v['gain'] for v in gains.values()):+.3f}}}")
        put("covPerturbMax", rf"\ensuremath{{{max(v['gain'] for v in gains.values()):+.3f}}}")
        spec = where_["specialty"]
        put("covSpecialtyMin", rf"\ensuremath{{{min(spec.values()):+.3f}}}")
        put("covSpecialtyMax", rf"\ensuremath{{{max(spec.values()):+.3f}}}")
        put("covSpecialtyMinName", min(spec, key=spec.__getitem__))
        put("covSpecialtyMaxName", max(spec, key=spec.__getitem__))
    expected = [
        f"covCov{token}{alias}{kind}"
        for *_, alias in PAIRS
        for token in [*CATEGORIES.values(), "All", "Nine"]
        for kind in ("Bare", "Prompt", "Diff")
    ]
    expected += [
        f"cov{token}{alias}{kind}"
        for *_, alias in PAIRS
        for token in ("Harm", "SevereHarm", "MildHarm")
        for kind in ("Bare", "Prompt", "Diff")
    ]
    expected += [f"cov{alias}{k}" for alias, *_ in RETEST for k in ("Items", "Agree", "Kappa")]
    expected += [
        f"cov{token}{alias}"
        for *_, alias in PAIRS
        for token in ("DBinary", "DRaw", "Mde", "P", "PHolm")
    ]
    expected += ["covMatchedDiff"]
    expected += [f"covPhys{token}" for token in PHYSICIAN_MACROS.values()]
    expected += [
        name
        for alias in ("Sonnet", "Terra", "Astra", "Grok")
        for token in ("Nemo", "Ling")
        for name in (
            f"covRatedF{alias}{token}",
            f"covRatedOk{alias}{token}Bare",
            f"covRatedOk{alias}{token}Prompt",
        )
    ]
    lines += [rf"\newcommand{{\{n}}}{{\covPending}}" for n in expected if n not in defined]
    return "\n".join(lines) + "\n"


def category_table(result: dict[str, Any]) -> str:
    """Rows of the manuscript's coverage table: a row per category, a column per model."""

    models = [alias for *_, alias in PAIRS if alias in result["mechanism"]]
    rows = []
    for category, token in [*CATEGORIES.items(), ("all", "All"), ("nine", "Nine")]:
        label = {
            "all": "All appropriate",
            "nine": "Score 9 (essential)",
            "Counseling": "Counselling",
        }.get(category, category)
        cells = [rf"\covCov{token}{alias}Bare$\to$\covCov{token}{alias}Prompt" for alias in models]
        rows.append(f"{label} & " + " & ".join(cells) + r"\\")
    harm = [rf"\covHarm{alias}Bare$\to$\covHarm{alias}Prompt" for alias in models]
    severe = [rf"\covSevereHarm{alias}Bare$\to$\covSevereHarm{alias}Prompt" for alias in models]
    rows.append(r"\midrule")
    rows.append("Harmful options proposed per item & " + " & ".join(harm) + r"\\")
    rows.append(r"\quad of which severe (score 1) & " + " & ".join(severe) + r"\\")
    names = {"Sonnet": "Sonnet 5.5", "Terra": "terra", "Grok": "Grok 4.7", "Muse": "Muse Spark"}
    header = " & " + " & ".join(names.get(m, m) for m in models) + r"\\"
    return (
        "\n".join(
            [
                r"\begin{tabular}{l" + "c" * len(models) + "}",
                r"\toprule",
                header,
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )
        + "\n"
    )


def figures(arms: dict[str, dict[str, dict[str, Any]]], result: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def mean_f1(name: str) -> float:
        return statistics.mean(case_mean(arms[name], "F1_weighted").values())

    def mean_actions(name: str) -> float:
        return statistics.mean(len(r["responseActions"]) for r in arms[name].values())

    # Figure 1: actions stated against F1. Every bare model; arrows to the instructed arm.
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for name in BARE_MODELS:
        if name in arms:
            ax.scatter(mean_actions(name), mean_f1(name), s=22, color=BARE, zorder=3, lw=0)
    for bare, _, prompted, _, alias in PAIRS:
        if bare not in arms or prompted not in arms:
            continue
        x0, y0 = mean_actions(bare), mean_f1(bare)
        x1, y1 = mean_actions(prompted), mean_f1(prompted)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1, "shrinkA": 4, "shrinkB": 4},
        )
        ax.scatter(x1, y1, s=26, color=PROMPT, zorder=4, lw=0)
        dx, dy = LABEL_OFFSETS.get(alias, (4, 2))
        ax.annotate(alias, (x1, y1), xytext=(dx, dy), textcoords="offset points", color=INK)
    ax.scatter([], [], s=22, color=BARE, label="bare case (12 models)")
    ax.scatter([], [], s=26, color=PROMPT, label="with instruction")
    ax.set_xlabel("Mean actions stated per plan")
    ax.set_ylabel("Severity-weighted F1")
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig-length-f1.pdf")
    plt.close(fig)

    # Figure 2: coverage of appropriate options by category, bare and instructed, per model.
    models = [alias for *_, alias in PAIRS if alias in result["mechanism"]]
    cats = list(CATEGORIES)
    fig, axes = plt.subplots(1, len(models), figsize=(7.2, 2.6), sharey=True)
    for ax, alias in zip(np.atleast_1d(axes), models, strict=True):
        row = result["mechanism"][alias]
        ypos = np.arange(len(cats))
        ax.barh(ypos + 0.2, [row[c]["bare"] for c in cats], height=0.36, color=BARE)
        ax.barh(ypos - 0.2, [row[c]["prompt"] for c in cats], height=0.36, color=PROMPT)
        ax.set_yticks(ypos, [c.replace("Counseling", "Counselling") for c in cats])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(alias, color=INK, fontsize=9)
        ax.grid(axis="x", color="#e6e5e1", lw=0.6)
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(color=BARE, label="bare case"),
            Patch(color=PROMPT, label="with instruction"),
        ],
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.supxlabel("Share of appropriate options covered", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "fig-categories.pdf")
    plt.close(fig)


def packet(arms: dict[str, dict[str, dict[str, Any]]], out: Path) -> dict[str, int]:
    """What a clinician should read: harmful options the instruction added, evaluator splits."""

    out.mkdir(parents=True, exist_ok=True)
    rubric = rubrics()
    added = []
    for bare, _, prompted, _, alias in PAIRS:
        if bare not in arms or prompted not in arms:
            continue
        for item, row in arms[prompted].items():
            options = rubric[case_of(item)]
            before = {
                int(o["id"])
                for o in arms[bare][item]["options"]
                if o.get("matched") or o.get("partial")
            }
            for verdict in row["options"]:
                option = int(verdict["id"])
                spec = options[option]
                credited = verdict.get("matched") or verdict.get("partial")
                if int(spec["score"]) <= 3 and credited and option not in before:
                    stated = [
                        a["action"]
                        for a in row["responseActions"]
                        if str(option) in str(a.get("match") or "").split(",")
                    ]
                    added.append(
                        {
                            "model": alias,
                            "item": item,
                            "specialist_score": spec["score"],
                            "category": spec["category"],
                            "rubric_option": spec["text"],
                            "plan_stated": " | ".join(stated),
                        }
                    )
    with (out / "harmful_options_added_by_instruction.csv").open("w", newline="") as sink:
        fields = [
            "model",
            "item",
            "specialist_score",
            "category",
            "rubric_option",
            "plan_stated",
        ]
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(added, key=lambda r: (r["specialist_score"], r["item"])))
    split: list[dict[str, Any]] = []
    for other in ("terra-specialist-jterra", "terra-specialist-jfree"):
        if other in arms and "terra-specialist" in arms:
            for item, row in arms["terra-specialist"].items():
                a = float(row["metrics"]["F1_weighted"])
                b = float(arms[other][item]["metrics"]["F1_weighted"])
                split.append({"evaluator": other, "item": item, "sonnet_f1": a, "other_f1": b})
    split.sort(key=lambda r: -abs(r["sonnet_f1"] - r["other_f1"]))
    with (out / "evaluator_disagreements.csv").open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=["evaluator", "item", "sonnet_f1", "other_f1"])
        writer.writeheader()
        writer.writerows(split[:40])
    (out / "README.md").write_text(
        "# Clinician review packet (simpledoctor-drky.26)\n\n"
        "Public NOHARM v2 cases (CC BY 4.0). Generated by scripts/noharm_paper_analyses.py; no"
        " model calls.\n\n"
        "- `harmful_options_added_by_instruction.csv`: options the specialists rated harmful"
        " (1-3) that an instructed plan proposed and the same model's bare plan did not, with"
        " the plan's own wording. Question for the reader: is the plan's action really the"
        " rubric option, and is it harmful for this patient?\n"
        "- `evaluator_disagreements.csv`: the items where two evaluators scored the same plan"
        " furthest apart. Question: which evaluator is right?\n"
        "- Off-rubric actions the audit raters flagged are in"
        " `datasets/eval/noharm_offrubric/<rater>/<arm>.jsonl`.\n"
    )
    return {"harmful_added": len(added), "disagreements": len(split[:40])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tex", type=Path)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--packet", type=Path)
    args = parser.parse_args()

    names = {(bare, bd) for bare, bd, *_ in PAIRS} | {(p, pd) for _, _, p, pd, _ in PAIRS}
    names |= {(a, "default") for a in ("terra-specialist-jterra", "terra-specialist-jfree")}
    import scripts.noharm_paper_snapshot as snapshot

    names |= {(name, directory) for name, (directory, _) in snapshot.ARMS.items()}
    arms = {}
    for name, directory in sorted(names):
        rows = judged(name, directory)
        if rows is not None:
            arms[name] = rows
    result = {
        "mechanism": mechanism(arms),
        "length_adjustment": length_adjustment(arms),
        "audit_adjusted": audit_adjusted(arms),
        "weighting": weighting(arms),
        "retest": retest(),
        "where": where(arms),
        "study": study_description(arms),
        "physician_review": physician_review(),
    }
    if args.tex:
        args.tex.write_text(macros(result), encoding="utf-8")
        args.tex.with_name("coverage-category-table.tex").write_text(
            category_table(result), encoding="utf-8"
        )
    if args.figures:
        figures(arms, result, args.figures)
    if args.packet:
        result["packet"] = packet(arms, args.packet)
    print(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
