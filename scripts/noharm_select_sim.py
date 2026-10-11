#!/usr/bin/env python
"""Which proposers and which selection rule would score best on NOHARM, before paying for them?

    .venv/bin/python scripts/noharm_select_sim.py --check muse-spark-1.3
    .venv/bin/python scripts/noharm_select_sim.py --arms muse-spark-1.3,big-pickle,gpt-5.6-terra \\
        --raters claude-sonnet-5-5,gpt-5.6-luna --reference muse-spark-1.3

Reads what is already on disk: every judged arm (each response action of a plan, with the rubric
options the judge matched it to) and every complete rater test (``scripts/noharm_rater.py``: a
model's 1-9 rating of every rubric option). A simulated system is, per item, the union of its
proposers' response actions — deduplicated by the set of options they matched — less the actions
a selection rule drops on the raters' ratings. Each item is then scored by the kit's own
``compute_harm`` and ``compute_metrics_for_case`` and aggregated as the kit aggregates (the mean
over a case's variants, then over cases), with a case-cluster bootstrap interval and a paired
difference against ``--reference``.

What this is and is not. Off-rubric actions are left out because the headline precision excludes
them, so they cannot move ``F1_weighted``. The ratings stand in for a live panel rating the
extracted actions, and the action-to-option mapping is the judge's, which a live system never
has — so the simulation **ranks designs; it does not predict a score**. ``--check`` reproduces an
arm's own ``F1_weighted`` from its actions, which is the test that the reconstruction is faithful.
A rule drops an action when any option it matched is flagged: one action matched to an indicated
and a harmful option is still the harmful action.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "sources/mast/benchmarks/donoharm"
JUDGED = ROOT / "sources/mast/results/raw/donoharm"
RATERS = ROOT / "datasets/eval/noharm_rater"
METRICS = ("F1_weighted", "Precision_weighted", "Recall_weighted", "Severe_rate")

sys.path.insert(0, str(KIT))
_kit_metrics: Any = importlib.import_module("judge.metrics")


@dataclass(frozen=True)
class Action:
    options: frozenset[int]
    sources: frozenset[str]


Rule = Callable[[Action, dict[str, list[int]]], bool]
"""Keep an action? Given it and each rater's ratings of the options it matched."""


def load_rubrics() -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted((KIT / "dataset/rubrics").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[data["id"]] = data
    return out


def load_arm(name: str) -> dict[str, dict[str, Any]]:
    paths = sorted(JUDGED.glob(f"*/{name}_judged.jsonl"))
    if len(paths) != 1:
        raise SystemExit(f"{name}: expected one judged file, found {len(paths)}")
    rows = [json.loads(x) for x in paths[0].read_text(encoding="utf-8").splitlines() if x.strip()]
    return {row["id"]: row for row in rows}


def load_rater(name: str) -> dict[tuple[str, int], int]:
    ratings: dict[tuple[str, int], int] = {}
    for line in (RATERS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line) if line.strip() else {}
        for option_id, score in (row.get("ratings") or {}).items():
            ratings[(row["case"], int(option_id))] = int(score)
    if len({case for case, _ in ratings}) != 30:
        raise SystemExit(f"rater {name} has not rated all 30 cases")
    return ratings


def option_ids(action: dict[str, Any]) -> frozenset[int]:
    return frozenset(int(x) for x in str(action.get("match") or "").split(",") if x.strip())


def score_item(
    rubric: dict[str, Any],
    matched: set[int],
    partial: set[int],
    actions: list[dict[str, Any]],
) -> dict[str, float]:
    judge_options = [
        {"id": o["id"], "matched": o["id"] in matched, "partial": o["id"] in partial}
        for o in rubric["options"]
    ]
    harm = _kit_metrics.compute_harm(judge_options, rubric)
    metrics = _kit_metrics.compute_metrics_for_case(harm, actions, rubric)
    return {name: float(metrics[name]) for name in METRICS}


def simulate(
    arms: dict[str, dict[str, dict[str, Any]]],
    proposers: Sequence[str],
    rule: Rule | None,
    raters: dict[str, dict[tuple[str, int], int]],
    rubrics: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Per item, the kit's metrics for the selected union of the proposers' actions."""

    items = sorted(set.intersection(*(set(arms[p]) for p in proposers)))
    out = {}
    for item in items:
        case = item.split("-")[0]
        merged: dict[frozenset[int], set[str]] = {}
        full: set[int] = set()
        for proposer in proposers:
            record = arms[proposer][item]
            full |= {o["id"] for o in record["options"] if o["matched"] and not o.get("partial")}
            for action in record["responseActions"]:
                ids = option_ids(action)
                if ids:
                    merged.setdefault(ids, set()).add(proposer)
        kept = [Action(ids, frozenset(sources)) for ids, sources in merged.items()]
        if rule is not None:
            kept = [
                a
                for a in kept
                if rule(
                    a, {r: [ratings[(case, o)] for o in a.options] for r, ratings in raters.items()}
                )
            ]
        matched = set().union(*(a.options for a in kept)) if kept else set()
        actions = [{"match": ",".join(map(str, sorted(a.options))), "score": 0} for a in kept]
        out[item] = score_item(rubrics[case], matched, matched - full, actions)
    return out


def check(arm: dict[str, dict[str, Any]], rubrics: dict[str, dict[str, Any]]) -> None:
    """An arm scored from its own actions and option verdicts must equal the kit's record."""

    worst = 0.0
    for item, record in arm.items():
        rubric = rubrics[item.split("-")[0]]
        matched = {o["id"] for o in record["options"] if o["matched"]}
        partial = {o["id"] for o in record["options"] if o["matched"] and o.get("partial")}
        mine = score_item(rubric, matched, partial, record["responseActions"])
        worst = max(worst, abs(mine["F1_weighted"] - record["metrics"]["F1_weighted"]))
    print(f"largest per-item F1_weighted difference from the kit's own record: {worst:.2e}")


def per_case(scores: dict[str, dict[str, float]], metric: str) -> dict[str, float]:
    by_case: dict[str, list[float]] = {}
    for item, metrics in scores.items():
        by_case.setdefault(item.split("-")[0], []).append(metrics[metric])
    return {case: sum(v) / len(v) for case, v in by_case.items()}


def interval(values: dict[str, float], draws: int = 4000, seed: int = 0) -> tuple[float, float]:
    cases = sorted(values)
    rng = random.Random(seed)
    means = sorted(sum(values[rng.choice(cases)] for _ in cases) / len(cases) for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


#: A pseudo-rater carrying the specialists' own scores, for ceilings only (``--oracle``).
ORACLE = "specialists"


def rules(primary: str | None, oracle: bool = False) -> dict[str, Rule | None]:
    def panel(scores: dict[str, list[int]]) -> list[list[int]]:
        return [s for name, s in scores.items() if name != ORACLE]

    def mean(scores: dict[str, list[int]]) -> float:
        per_rater = [min(s) for s in panel(scores)]
        return sum(per_rater) / len(per_rater)

    def lowest(scores: dict[str, list[int]]) -> int:
        return min(min(s) for s in panel(scores))

    def scrutinise_consensus(action: Action, scores: dict[str, list[int]]) -> bool:
        """Plausibility-aware: what several proposers agree on is held to the stricter veto,
        because the harms raters miss most are the ones many models propose."""

        return lowest(scores) > (3 if len(action.sources) >= 2 else 2)

    def trust_consensus(action: Action, scores: dict[str, list[int]]) -> bool:
        """Majority-vote intuition, the opposite: agreement relaxes the veto."""

        return lowest(scores) > (1 if len(action.sources) >= 2 else 3)

    def grs(action: Action, scores: dict[str, list[int]]) -> bool:
        if lowest(scores) <= 2:
            return False
        if mean(scores) >= 7:
            return True
        return mean(scores) > 3 and (primary in action.sources or len(action.sources) >= 2)

    table: dict[str, Rule | None] = {
        "none": None,
        "veto<=3": lambda a, s: lowest(s) > 3,
        "veto<=2": lambda a, s: lowest(s) > 2,
        "mean<=3": lambda a, s: mean(s) > 3,
        "scrutinise-consensus": scrutinise_consensus,
        "trust-consensus": trust_consensus,
    }
    if oracle:
        table["ORACLE(drop specialist 1-3)"] = lambda a, s: min(s[ORACLE]) >= 4
    if primary is not None:
        table[f"grs(primary {primary})"] = grs
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", help="reproduce one arm's own F1_weighted from its actions")
    parser.add_argument("--arms", help="pool of judged arms, comma-separated")
    parser.add_argument("--raters", help="complete rater tests, comma-separated")
    parser.add_argument("--reference", help="arm the paired differences are taken against")
    parser.add_argument("--max-proposers", type=int, default=3)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--oracle", action="store_true", help="add the specialists' own scores as a ceiling rule"
    )
    args = parser.parse_args(argv)

    rubrics = load_rubrics()
    if args.check:
        check(load_arm(args.check), rubrics)
        return 0
    if not (args.arms and args.raters):
        raise SystemExit("--arms and --raters are required (or --check)")
    pool = args.arms.split(",")
    arms = {name: load_arm(name) for name in pool}
    panel = {name: load_rater(name) for name in args.raters.split(",")}
    if args.oracle:
        panel[ORACLE] = {
            (case, int(o["id"])): int(o["score"])
            for case, rubric in rubrics.items()
            for o in rubric["options"]
        }
    results: list[tuple[str, dict[str, dict[str, float]]]] = []
    for size in range(1, min(args.max_proposers, len(pool)) + 1):
        for proposers in itertools.combinations(pool, size):
            primaries: list[str | None] = [None, *proposers] if size > 1 else [None]
            seen: set[str] = set()
            for primary in primaries:
                for rule_name, rule in rules(primary, args.oracle).items():
                    label = f"{'+'.join(proposers)} | {rule_name}"
                    if label in seen:
                        continue
                    seen.add(label)
                    results.append((label, simulate(arms, proposers, rule, panel, rubrics)))

    reference = None
    if args.reference:
        reference = per_case(simulate(arms, [args.reference], None, panel, rubrics), "F1_weighted")
    ranked = sorted(
        results,
        key=lambda row: -sum(per_case(row[1], "F1_weighted").values()),
    )
    print(f"raters: {', '.join(panel)}; items: {len(ranked[0][1]) if ranked else 0}")
    for label, scores in ranked[: args.top]:
        f1 = per_case(scores, "F1_weighted")
        low, high = interval(f1)
        cells = [f"F1 {sum(f1.values()) / len(f1):.3f} [{low:.3f}, {high:.3f}]"]
        for metric, short in (("Precision_weighted", "P"), ("Recall_weighted", "R")):
            values = per_case(scores, metric)
            finite = [v for v in values.values() if not math.isnan(v)]
            cells.append(f"{short} {sum(finite) / len(finite):.3f}")
        cells.append(f"severe {sum(per_case(scores, 'Severe_rate').values()) / len(f1):.3f}")
        if reference is not None:
            diff = {case: f1[case] - reference[case] for case in f1 if case in reference}
            d_low, d_high = interval(diff)
            cells.append(f"Δ {sum(diff.values()) / len(diff):+.3f} [{d_low:+.3f}, {d_high:+.3f}]")
        print(f"{label:70s} " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
