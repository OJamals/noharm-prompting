"""Recompute the prompting paper's results without any model or network calls.

Prints a JSON evidence snapshot and, with ``--tex``, writes every number the manuscript quotes as a
LaTeX macro, so no figure is transcribed by hand. Source files are hashed; an incomplete arm is
refused, or with ``--partial`` listed under ``pending_arms`` and left out of every comparison that
needs it. All intervals use the same unstratified case-cluster bootstrap (10,000 draws, seed 7).

Beside the kit's headline ``F1_weighted`` each arm reports what that metric cannot see
(simpledoctor-drky.26.9): the kit's off-rubric-included precision (``Precision_all``) folded into an
F1, the share of stated actions that match no rubric option (the kit judges none of them), the
worst of a case's 11 variants, and how long the plan is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts import noharm_offrubric_audit as audit
from scripts.noharm_select_sim import interval, load_rubrics, per_case, simulate

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "sources/mast/results/raw/donoharm"
METRICS = ("F1_weighted", "Precision_weighted", "Recall_weighted", "Severe_rate")
DRAWS, SEED = 10000, 7

#: arm -> (kit prompt directory, macro alias). The alias names the arm in ``--tex`` output; LaTeX
#: command names take letters only.
ARMS: dict[str, tuple[str, str]] = {
    # Bare case, as MAST's unprompted protocol sends it.
    "claude-sonnet-5-5": ("default", "SonnetBare"),
    "gpt-5.6-terra": ("default", "TerraBare"),
    "gpt-6-astra": ("default", "AstraBare"),
    "grok-4.7": ("default", "GrokBare"),
    "gpt-5.6-luna": ("default", "LunaBare"),
    "claude-opus-5-5": ("default", "OpusBare"),
    "gpt-6.1-sol": ("default", "SolBare"),
    "claude-haiku-4-5-20251001": ("default", "HaikuBare"),
    "claude-sonnet-5": ("default", "SonnetOldBare"),
    "muse-spark-1.3": ("default", "MuseBare"),
    "nemotron-3-ultra": ("default", "NemotronBare"),
    "big-pickle": ("default", "PickleBare"),
    # One prompt for every model: consult framing + plan format (``specialist``).
    "sonnet-specialist": ("default", "SonnetSpec"),
    "terra-specialist": ("default", "TerraSpec"),
    "astra-specialist": ("default", "AstraSpec"),
    "grok-specialist": ("default", "GrokSpec"),
    "luna-specialist": ("default", "LunaSpec"),
    "opus-specialist": ("default", "OpusSpec"),
    "sol-specialist": ("default", "SolSpec"),
    "haiku-specialist": ("default", "HaikuSpec"),
    "sonnet5-specialist": ("default", "SonnetOldSpec"),
    "muse-specialist": ("default", "MuseSpec"),
    "nemotron-specialist": ("default", "NemotronSpec"),
    "pickle-specialist": ("default", "PickleSpec"),
    # Other prompts.
    "sonnet-pharmacist": ("default", "SonnetPharm"),
    "sonnet-pcp": ("default", "SonnetPcp"),
    "astra-exhaustive": ("default", "AstraExh"),
    "menu-terra": ("default", "TerraMenu"),
    "grok-4.7-thorough": ("thorough", "GrokThorough"),
    "muse-spark-1.3-thorough": ("thorough", "MuseThorough"),
    "sonnet-thorough": ("thorough", "SonnetThorough"),
    # Components of the specialist prompt, and a length cap (drky.26.4, .5).
    "sonnet-format": ("default", "SonnetFormat"),
    "sonnet-framing": ("default", "SonnetFraming"),
    "sonnet-length": ("default", "SonnetLength"),
    "sonnet-exhaustive": ("default", "SonnetExh"),
    "sonnet-specialist500": ("default", "SonnetCap"),
    "terra-format": ("default", "TerraFormat"),
    "terra-framing": ("default", "TerraFraming"),
    "terra-length": ("default", "TerraLength"),
    "terra-exhaustive": ("default", "TerraExh"),
    "terra-specialist500": ("default", "TerraCap"),
    # Noise: the same prompt regenerated, the same plans re-judged (drky.26.6).
    "terra-specialist-r2": ("default", "TerraSpecRtwo"),
    "terra-specialist-r3": ("default", "TerraSpecRthree"),
    "gpt-5.6-terra-r2": ("default", "TerraBareRtwo"),
    "terra-specialist-rj": ("default", "TerraSpecRejudge"),
    # Systems built from several calls.
    "concat3-eligible": ("default", "UnionBare"),
    "concat4-eligible-gap": ("default", "Gap"),
    "concat4-eligible-gapopus": ("default", "GapOpus"),
    "write-eligible3": ("default", "WriterBare"),
    "concat3-prompted": ("default", "UnionPrompt"),
    "concat2-terraspec-gap": ("default", "TerraSpecGap"),
    "selfmoa-terra3": ("default", "SelfMoa"),
    # The same plans under another judge family.
    "claude-sonnet-5-5-jterra": ("default", "SonnetBareJT"),
    "sonnet-pcp-jterra": ("default", "SonnetPcpJT"),
    "gpt-5.6-terra-jterra": ("default", "TerraBareJT"),
    "terra-specialist-jterra": ("default", "TerraSpecJT"),
    "concat3-prompted-jterra": ("default", "UnionPromptJT"),
    "muse-spark-1.3-jterra": ("default", "MuseBareJT"),
    "gpt-5.6-terra-jgrok": ("default", "TerraBareJG"),
    "terra-specialist-jgrok": ("default", "TerraSpecJG"),
    # A free model from a family no arm comes from, chosen by its agreement with the kit's
    # Gemini-judged example (drky.26.1; paid judging paused 2026-10-10).
    "gpt-5.6-terra-jfree": ("default", "TerraBareJF"),
    "terra-specialist-jfree": ("default", "TerraSpecJF"),
    "claude-sonnet-5-5-jfree": ("default", "SonnetBareJF"),
    "sonnet-specialist-jfree": ("default", "SonnetSpecJF"),
}

#: A re-judged arm and the arm whose plans it must carry unchanged.
SAME_PLANS = {
    name: name.rsplit("-", 1)[0]
    for name in ARMS
    if name.endswith(("-jterra", "-jgrok", "-jfree", "-rj")) and not name.startswith("gpt55")
}

#: bare model -> its specialist arm, for the grid (drky.26.3).
GRID = {
    "claude-sonnet-5-5": "sonnet-specialist",
    "gpt-5.6-terra": "terra-specialist",
    "gpt-6-astra": "astra-specialist",
    "grok-4.7": "grok-specialist",
    "gpt-5.6-luna": "luna-specialist",
    "claude-opus-5-5": "opus-specialist",
    "gpt-6.1-sol": "sol-specialist",
    "claude-haiku-4-5-20251001": "haiku-specialist",
    "claude-sonnet-5": "sonnet5-specialist",
    "muse-spark-1.3": "muse-specialist",
    "nemotron-3-ultra": "nemotron-specialist",
    "big-pickle": "pickle-specialist",
}

#: Each model's prompted arm as it was actually run (drky.26 paused before the uniform grid
#: finished): the uncapped specialist-style instruction for three models, the kit's 500-word
#: thorough prompt for two.
PAPER_GRID = {
    "claude-sonnet-5-5": "sonnet-specialist",
    "gpt-5.6-terra": "terra-specialist",
    "gpt-6-astra": "astra-exhaustive",
    "grok-4.7": "grok-4.7-thorough",
    "muse-spark-1.3": "muse-spark-1.3-thorough",
}
UNCAPPED = ("claude-sonnet-5-5", "gpt-5.6-terra", "gpt-6-astra")

#: Single-call arms whose length and score enter the across-configuration correlation.
SINGLE_CALL = tuple(
    name
    for name in ARMS
    if not name.startswith(("concat", "write", "selfmoa", "gap"))
    and not name.endswith(("-jterra", "-jgrok", "-jfree", "-rj", "-r2", "-r3"))
)

#: (a, b, macro alias): the paired difference b - a, per base case.
PAIRS: tuple[tuple[str, str, str], ...] = (
    *((bare, spec, f"Prompt{ARMS[bare][1].removesuffix('Bare')}") for bare, spec in GRID.items()),
    ("claude-sonnet-5-5", "sonnet-pharmacist", "PromptSonnetPharm"),
    ("claude-sonnet-5-5", "sonnet-pcp", "PromptSonnetPcp"),
    ("gpt-6-astra", "astra-exhaustive", "PromptAstraExh"),
    ("gpt-5.6-terra", "menu-terra", "PromptTerraMenu"),
    ("grok-4.7", "grok-4.7-thorough", "PromptGrokThorough"),
    ("muse-spark-1.3", "muse-spark-1.3-thorough", "PromptMuseThorough"),
    ("claude-sonnet-5-5", "sonnet-thorough", "PromptSonnetThorough"),
    ("sonnet-pharmacist", "sonnet-pcp", "PcpVsPharm"),
    ("sonnet-specialist", "sonnet-pcp", "PcpVsSpec"),
    # Components: each against the bare case, and against the full specialist prompt.
    *(
        (base, f"{stem}-{part}", f"{alias}{part.capitalize()}VsBare")
        for stem, base, alias in (
            ("sonnet", "claude-sonnet-5-5", "Sonnet"),
            ("terra", "gpt-5.6-terra", "Terra"),
        )
        for part in ("format", "framing", "length", "exhaustive")
    ),
    *(
        (spec, f"{stem}-{part}", f"{alias}{part.capitalize()}VsSpec")
        for stem, spec, alias in (
            ("sonnet", "sonnet-specialist", "Sonnet"),
            ("terra", "terra-specialist", "Terra"),
        )
        for part in ("format", "framing", "length", "exhaustive")
    ),
    ("sonnet-specialist", "sonnet-specialist500", "SonnetCapVsSpec"),
    ("terra-specialist", "terra-specialist500", "TerraCapVsSpec"),
    # Noise.
    ("terra-specialist", "terra-specialist-r2", "TerraSpecRtwo"),
    ("terra-specialist", "terra-specialist-r3", "TerraSpecRthree"),
    ("gpt-5.6-terra", "gpt-5.6-terra-r2", "TerraBareRtwo"),
    ("terra-specialist", "terra-specialist-rj", "TerraSpecRejudge"),
    # One prompted call against systems of several calls.
    ("claude-sonnet-5-5", "concat3-eligible", "UnionBareVsSonnet"),
    ("concat3-eligible", "concat4-eligible-gap", "GapVsUnion"),
    ("concat3-eligible", "write-eligible3", "WriterVsUnion"),
    ("concat4-eligible-gap", "concat4-eligible-gapopus", "OpusVsSonnetGap"),
    ("concat4-eligible-gap", "terra-specialist", "TerraSpecVsGap"),
    ("terra-specialist", "concat3-prompted", "UnionPromptVsTerraSpec"),
    ("terra-specialist", "concat2-terraspec-gap", "TerraSpecGapVsSpec"),
    ("terra-specialist", "selfmoa-terra3", "SelfMoaVsTerraSpec"),
    ("selfmoa-terra3", "concat3-prompted", "UnionPromptVsSelfMoa"),
    # The prompting gain and the ensemble gain under another judge family.
    ("gpt-5.6-terra-jterra", "terra-specialist-jterra", "PromptTerraJT"),
    ("claude-sonnet-5-5-jterra", "sonnet-pcp-jterra", "PromptSonnetPcpJT"),
    ("gpt-5.6-terra-jgrok", "terra-specialist-jgrok", "PromptTerraJG"),
    ("gpt-5.6-terra-jfree", "terra-specialist-jfree", "PromptTerraJF"),
    ("claude-sonnet-5-5-jfree", "sonnet-specialist-jfree", "PromptSonnetJF"),
    ("terra-specialist-jterra", "concat3-prompted-jterra", "UnionPromptVsTerraSpecJT"),
)

#: grid_summary key -> macro name
GRID_NAMES = {
    "rho_gain_bare_actions": "RhoGainActions",
    "rho_gain_bare_f1": "RhoGainBare",
    "sd_bare": "SdBare",
    "sd_spec": "SdSpec",
    "sd_reduction": "SdReduction",
    "range_bare": "RangeBare",
    "range_spec": "RangeSpec",
    "kendall_bare_spec": "Kendall",
    "mean_gain": "MeanGain",
}

#: Judge x prompt interactions: (bare, prompted) under Sonnet against the same plans re-judged.
INTERACTIONS = (
    ("gpt-5.6-terra", "terra-specialist", "-jterra", "TerraJT"),
    ("gpt-5.6-terra", "terra-specialist", "-jgrok", "TerraJG"),
    ("claude-sonnet-5-5", "sonnet-pcp", "-jterra", "SonnetPcpJT"),
    ("gpt-5.6-terra", "terra-specialist", "-jfree", "TerraJF"),
    ("claude-sonnet-5-5", "sonnet-specialist", "-jfree", "SonnetJF"),
)


def source(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def rows(path: Path) -> dict[str, dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indexed = {record["id"]: record for record in records}
    if len(records) != 330 or len(indexed) != 330:
        raise ValueError(f"{path.name}: expected 330 distinct completed variants")
    counts: dict[str, int] = {}
    for item in indexed:
        case = item.split("-")[0]
        counts[case] = counts.get(case, 0) + 1
    if len(counts) != 30 or set(counts.values()) != {11}:
        raise ValueError(f"{path.name}: expected 30 cases with 11 variants each")
    return indexed


def estimate(values: dict[str, float]) -> dict[str, Any]:
    low, high = interval(values, draws=DRAWS, seed=SEED)
    return {"mean": statistics.mean(values.values()), "ci95": [low, high], "cases": len(values)}


def case_scores(scores: dict[str, dict[str, Any]], metric: str) -> dict[str, float]:
    """Match the kit's nanmean: precision is undefined when no action matches the rubric."""

    by_case: dict[str, list[float]] = {}
    for item, result in scores.items():
        value = float(result[metric])
        if math.isfinite(value):
            by_case.setdefault(item.split("-")[0], []).append(value)
    return {case: statistics.mean(values) for case, values in by_case.items()}


def f1_all(metrics: dict[str, Any]) -> float:
    """F1 with the kit's off-rubric-included precision: a stated action no option matches counts."""

    precision, recall = float(metrics["Precision_all"]), float(metrics["Recall_weighted"])
    if not math.isfinite(precision):
        return math.nan
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def worst_variant(scores: dict[str, dict[str, Any]]) -> dict[str, float]:
    by_case: dict[str, list[float]] = {}
    for item, result in scores.items():
        by_case.setdefault(item.split("-")[0], []).append(float(result["F1_weighted"]))
    return {case: min(values) for case, values in by_case.items()}


def output_tokens(record: dict[str, Any]) -> float | None:
    usage = record.get("usage") or {}
    for key in ("completion_tokens", "output_tokens"):
        if isinstance(usage.get(key), int | float):
            return float(usage[key])
    return None


def runtime_quantiles(records: Any) -> dict[str, float] | None:
    """Median and 90th percentile seconds per response, where the generator recorded them."""

    times = sorted(
        float(r["runtime"]) for r in records if isinstance(r.get("runtime"), int | float)
    )
    if len(times) < 330:
        return None
    return {"median": statistics.median(times), "p90": times[int(0.9 * len(times))]}


def revision(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def complete(name: str) -> bool:
    directory = ARMS[name][0]
    try:
        rows(DATA / directory / f"{name}_judged.jsonl")
        raw = rows(DATA / directory / f"{name}.jsonl")
    except (OSError, ValueError):
        return False
    return not any(r.get("error") or len(r.get("response") or "") < 50 for r in raw.values())


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=values.__getitem__)
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out

    return statistics.correlation(ranks(x), ranks(y))


def kendall(x: list[float], y: list[float]) -> float:
    concordant = discordant = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            s = (x[i] - x[j]) * (y[i] - y[j])
            concordant += s > 0
            discordant += s < 0
    pairs = len(x) * (len(x) - 1) / 2
    return (concordant - discordant) / pairs


def grid_summary(
    f1: dict[str, dict[str, float]],
    actions: dict[str, dict[str, float]],
    grid: dict[str, str],
    minimum: int = 4,
) -> dict[str, Any] | None:
    """Across models with both arms: gain against bare length, spread, and rank agreement."""

    models = [bare for bare, spec in grid.items() if bare in f1 and spec in f1]
    if len(models) < minimum:
        return None
    cases = sorted(f1[models[0]])

    def at(selection: list[str]) -> dict[str, float]:
        bare = [statistics.mean(f1[m][c] for c in selection) for m in models]
        spec = [statistics.mean(f1[grid[m]][c] for c in selection) for m in models]
        length = [statistics.mean(actions[m][c] for c in selection) for m in models]
        gain = [s - b for s, b in zip(spec, bare, strict=True)]
        return {
            "rho_gain_bare_actions": spearman(length, gain),
            "rho_gain_bare_f1": spearman(bare, gain),
            "sd_bare": statistics.stdev(bare),
            "sd_spec": statistics.stdev(spec),
            "sd_reduction": statistics.stdev(bare) - statistics.stdev(spec),
            "range_bare": max(bare) - min(bare),
            "range_spec": max(spec) - min(spec),
            "kendall_bare_spec": kendall(bare, spec),
            "mean_gain": statistics.mean(gain),
        }

    point = at(cases)
    rng = random.Random(SEED)
    draws = [at([rng.choice(cases) for _ in cases]) for _ in range(DRAWS)]

    def ci(key: str) -> list[float]:
        values = sorted(draw[key] for draw in draws)
        return [values[int(0.025 * DRAWS)], values[int(0.975 * DRAWS) - 1]]

    return {
        "models": models,
        **{key: {"value": value, "ci95": ci(key)} for key, value in point.items()},
    }


def length_correlation(
    f1: dict[str, dict[str, float]], actions: dict[str, dict[str, float]]
) -> dict[str, Any] | None:
    """Spearman between an arm's mean stated actions and its F1, over every single-call arm."""

    arms = [name for name in SINGLE_CALL if name in f1]
    if len(arms) < 5:
        return None
    cases = sorted(f1[arms[0]])

    def at(selection: list[str]) -> float:
        length = [statistics.mean(actions[a][c] for c in selection) for a in arms]
        score = [statistics.mean(f1[a][c] for c in selection) for a in arms]
        return spearman(length, score)

    rng = random.Random(SEED)
    draws = sorted(at([rng.choice(cases) for _ in cases]) for _ in range(DRAWS))
    return {
        "arms": arms,
        "rho": at(cases),
        "ci95": [draws[int(0.025 * DRAWS)], draws[int(0.975 * DRAWS) - 1]],
    }


#: bare arm -> (prompted arm, macro alias) for the off-rubric audit (drky.26.2).
AUDIT_PAIRS = {
    "claude-sonnet-5-5": ("sonnet-pcp", "Sonnet"),
    "gpt-5.6-terra": ("terra-specialist", "Terra"),
    "gpt-6-astra": ("astra-exhaustive", "Astra"),
    "grok-4.7": ("grok-4.7-thorough", "Grok"),
}


def offrubric_audit() -> dict[str, Any]:
    """The audit, read from its caches: raters that rated every sampled item of both arms."""

    sample = len(audit.sample_ids())
    raters = (
        sorted(
            p.name.replace("_", "/", 1) if p.name.startswith("opencode_") else p.name
            for p in audit.OUT.iterdir()
            if p.is_dir()
        )
        if audit.OUT.exists()
        else []
    )
    out: dict[str, Any] = {"sample_items": sample, "pairs": {}}
    for bare, (prompted, alias) in AUDIT_PAIRS.items():
        complete = [
            r
            for r in raters
            if len(audit.load(r, bare)) == sample and len(audit.load(r, prompted)) == sample
        ]
        if not complete:
            continue
        rows = audit.report(complete, {bare: prompted})["pairs"]
        # Nemotron rated every pair, so it is the primary view wherever it is complete; a second
        # rater adds its own view, the both-agree view and the agreement between the two.
        primary = next((r for r in complete if "nemotron" in r), complete[0])
        entry: dict[str, Any] = {
            "raters": complete,
            "primary": primary,
            "views": {rater_token(row["rater"]): row for row in rows},
        }
        if len(complete) == 2:
            pooled = [
                {
                    f"{arm}/{item}": row
                    for arm in (bare, prompted)
                    for item, row in audit.load(r, arm).items()
                }
                for r in complete
            ]
            entry["agreement"] = audit.agreement(pooled)
        out["pairs"][alias] = entry
    return out


def rater_token(rater: str) -> str:
    """A macro suffix for an audit view: the rater's family, or ``Both`` for the joint view."""

    if rater == "both":
        return "Both"
    for needle, token in (("nemotron", "Nemo"), ("ling", "Ling"), ("luna", "Luna")):
        if needle in rater:
            return token
    raise ValueError(f"no macro token for rater {rater!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true", help="skip arms not yet complete")
    parser.add_argument("--tex", type=Path, help="write the manuscript's macros here")
    args = parser.parse_args()

    rubrics = load_rubrics()
    snapshot: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_status": "exploratory development-set analysis; not clinical validation",
        "submission_ready": False,
        "human_verification": "pending",
        "repo_revision": revision(ROOT),
        "kit_revision": revision(ROOT / "sources/mast"),
        "cases": 30,
        "variants": 330,
        "bootstrap": {"unit": "base case", "stratified": False, "draws": DRAWS, "seed": SEED},
        "arms": {},
        "pending_arms": [],
        "comparisons": [],
        "sources": [],
    }
    names = [name for name in ARMS if not args.partial or complete(name)]
    snapshot["pending_arms"] = [name for name in ARMS if name not in names]
    vectors: dict[str, dict[str, dict[str, float]]] = {
        "original": {},
        "deduplicated": {},
        "all_actions": {},
    }
    lengths: dict[str, dict[str, float]] = {}
    raw: dict[str, dict[str, dict[str, Any]]] = {}
    for name in names:
        directory = ARMS[name][0]
        judged_path = DATA / directory / f"{name}_judged.jsonl"
        raw_path = DATA / directory / f"{name}.jsonl"
        judged = rows(judged_path)
        raw[name] = rows(raw_path)
        if any(r.get("error") or len(r.get("response") or "") < 50 for r in raw[name].values()):
            raise ValueError(f"{name}: generation is incomplete")
        original = {item: r["metrics"] for item, r in judged.items()}
        dedup = simulate({name: judged}, [name], None, {}, rubrics)
        results = {}
        for mode, scores in (("original", original), ("deduplicated", dedup)):
            results[mode] = {metric: estimate(case_scores(scores, metric)) for metric in METRICS}
            vectors[mode][name] = per_case(scores, "F1_weighted")
        verbosity = {item: {"F1_all": f1_all(m)} for item, m in original.items()}
        vectors["all_actions"][name] = case_scores(verbosity, "F1_all")
        results["verbosity"] = {
            "F1_all": estimate(case_scores(verbosity, "F1_all")),
            "Offrubric_rate": estimate(case_scores(original, "Offrubric_rate")),
            "worst_variant_F1": estimate(worst_variant(original)),
        }
        lengths[name] = per_case(
            {item: {"n": float(len(r["responseActions"]))} for item, r in judged.items()}, "n"
        )
        tokens = [t for t in map(output_tokens, raw[name].values()) if t is not None]
        snapshot["arms"][name] = {
            "judge": sorted({str(r["judge"]) for r in judged.values()}),
            "median_extracted_actions": statistics.median(
                len(r["responseActions"]) for r in judged.values()
            ),
            "median_response_chars": statistics.median(
                len(r["response"]) for r in raw[name].values()
            ),
            "median_output_tokens": statistics.median(tokens) if len(tokens) == 330 else None,
            "runtime_s": runtime_quantiles(raw[name].values()),
            "results": results,
        }
        snapshot["sources"].extend([source(judged_path), source(raw_path)])
    for a, b, alias in PAIRS:
        if a not in raw or b not in raw:
            continue
        for mode in vectors:
            left, right = vectors[mode][a], vectors[mode][b]
            if set(left) != set(right):
                raise ValueError(f"{a}, {b}: unpaired case sets")
            result = estimate({case: right[case] - left[case] for case in sorted(left)})
            snapshot["comparisons"].append({"a": a, "b": b, "alias": alias, "mode": mode, **result})
    for name, plans in SAME_PLANS.items():
        if (
            name in raw
            and plans in raw
            and any(raw[name][i]["response"] != raw[plans][i]["response"] for i in raw[plans])
        ):
            raise ValueError(f"{name}: re-judging changed the underlying plans")
    snapshot["frozen_rejudge_plans_identical"] = True
    original = vectors["original"]
    if {"claude-sonnet-5-5", "muse-spark-1.3"} <= set(raw) and {
        "claude-sonnet-5-5-jterra",
        "muse-spark-1.3-jterra",
    } <= set(raw):
        snapshot["judge_arm_interaction"] = estimate(
            {
                case: (original["claude-sonnet-5-5"][case] - original["muse-spark-1.3"][case])
                - (
                    original["claude-sonnet-5-5-jterra"][case]
                    - original["muse-spark-1.3-jterra"][case]
                )
                for case in original["claude-sonnet-5-5"]
            }
        )
    snapshot["judge_prompt_interactions"] = {}
    for bare, prompted, suffix, alias in INTERACTIONS:
        needed = {bare, prompted, bare + suffix, prompted + suffix}
        if needed <= set(raw):
            snapshot["judge_prompt_interactions"][alias] = estimate(
                {
                    case: (original[prompted][case] - original[bare][case])
                    - (original[prompted + suffix][case] - original[bare + suffix][case])
                    for case in original[bare]
                }
            )
    snapshot["grid"] = grid_summary(original, lengths, GRID)
    snapshot["paper_grid"] = grid_summary(original, lengths, PAPER_GRID)
    snapshot["uncapped_grid"] = grid_summary(
        original, lengths, {m: PAPER_GRID[m] for m in UNCAPPED}, minimum=3
    )
    snapshot["length_correlation"] = length_correlation(original, lengths)
    snapshot["offrubric_audit"] = offrubric_audit()
    if "write-eligible3" in raw:
        trace = [r.get("usage", {}) for r in raw["write-eligible3"].values()]
        if all("missing_by_round" in row and "actions" in row for row in trace):
            first = [r["missing_by_round"][0] for r in trace]
            last = [r["missing_by_round"][-1] for r in trace]
            snapshot["writer_coverage_trace"] = {
                "source_actions_median": statistics.median(r["actions"] for r in trace),
                "missing_initial_median": statistics.median(first),
                "missing_initial_mean": statistics.mean(first),
                "missing_final_median": statistics.median(last),
                "missing_final_mean": statistics.mean(last),
            }
    for path in [
        Path(__file__),
        ROOT / "scripts/noharm_system.py",
        ROOT / "scripts/noharm_kit.py",
        ROOT / "scripts/noharm_select_sim.py",
        ROOT / "sources/mast/benchmarks/donoharm/judge/metrics.py",
        ROOT / "sources/mast/benchmarks/donoharm/DATA.md",
    ]:
        snapshot["sources"].append(source(path))
    for path in sorted((ROOT / "sources/mast/benchmarks/donoharm/dataset/rubrics").glob("*.json")):
        snapshot["sources"].append(source(path))
    if args.tex:
        args.tex.write_text(macros(snapshot), encoding="utf-8")
    print(json.dumps(snapshot, indent=2, allow_nan=False))


def macros(snapshot: dict[str, Any]) -> str:
    """Every quoted number as a macro: cov<Metric><Alias>, covD<Alias>, covGrid<Stat>."""

    def num(value: float, digits: int = 3) -> str:
        return f"{value:.{digits}f}"

    def ci(pair: list[float], digits: int = 3) -> str:
        return rf"\ensuremath{{[{num(pair[0], digits)},\,{num(pair[1], digits)}]}}"

    def difference(row: dict[str, Any]) -> str:
        low, high = row["ci95"]
        return rf"\ensuremath{{{row['mean']:+.3f}\,[{low:+.3f},\,{high:+.3f}]}}"

    lines = [
        "% Generated by scripts/noharm_paper_snapshot.py --tex; do not edit by hand.",
        f"% {snapshot['generated_at_utc']} repo {snapshot['repo_revision'][:12]}",
        f"% pending arms: {', '.join(snapshot['pending_arms']) or 'none'}",
        r"\providecommand{\covPending}{\textcolor{gray}{pending}}",
    ]
    for name, arm in snapshot["arms"].items():
        alias = ARMS[name][1]
        result = arm["results"]
        original, dedup, verbosity = result["original"], result["deduplicated"], result["verbosity"]
        values = {
            "F": num(original["F1_weighted"]["mean"]),
            "CI": ci(original["F1_weighted"]["ci95"]),
            "P": num(original["Precision_weighted"]["mean"]),
            "R": num(original["Recall_weighted"]["mean"]),
            "S": num(original["Severe_rate"]["mean"]),
            "A": f"{arm['median_extracted_actions']:g}",
            "FDedup": num(dedup["F1_weighted"]["mean"]),
            "CIDedup": ci(dedup["F1_weighted"]["ci95"]),
            "FAll": num(verbosity["F1_all"]["mean"]),
            "Off": f"{100 * verbosity['Offrubric_rate']['mean']:.0f}",
            "Worst": num(verbosity["worst_variant_F1"]["mean"]),
            "Chars": f"{arm['median_response_chars']:,.0f}".replace(",", "{,}"),
        }
        if arm.get("runtime_s"):
            values["Runtime"] = f"{arm['runtime_s']['median']:.0f}"
            values["RuntimePninety"] = f"{arm['runtime_s']['p90']:.0f}"
        for key, value in values.items():
            lines.append(rf"\newcommand{{\cov{key}{alias}}}{{{value}}}")
    for row in snapshot["comparisons"]:
        prefix = {"original": "covD", "deduplicated": "covDDedup", "all_actions": "covDAll"}[
            row["mode"]
        ]
        text = difference(row)
        lines.append(rf"\newcommand{{\{prefix}{row['alias']}}}{{{text}}}")
    for alias, row in snapshot.get("judge_prompt_interactions", {}).items():
        text = difference(row)
        lines.append(rf"\newcommand{{\covInteraction{alias}}}{{{text}}}")
    if "judge_arm_interaction" in snapshot:
        row = snapshot["judge_arm_interaction"]
        text = difference(row)
        lines.append(rf"\newcommand{{\covJudgeInteraction}}{{{text}}}")
    grid = snapshot.get("grid")
    if grid:
        lines.append(rf"\newcommand{{\covGridModels}}{{{len(grid['models'])}}}")
        for key, alias in GRID_NAMES.items():
            lines.append(rf"\newcommand{{\covGrid{alias}}}{{{num(grid[key]['value'], 2)}}}")
            lines.append(rf"\newcommand{{\covGrid{alias}CI}}{{{ci(grid[key]['ci95'], 2)}}}")
    for section, prefix in (("paper_grid", "covPaperGrid"), ("uncapped_grid", "covUncappedGrid")):
        summary = snapshot.get(section)
        if summary:
            lines.append(rf"\newcommand{{\{prefix}Models}}{{{len(summary['models'])}}}")
            for key, alias in GRID_NAMES.items():
                lines.append(rf"\newcommand{{\{prefix}{alias}}}{{{num(summary[key]['value'], 2)}}}")
                lines.append(rf"\newcommand{{\{prefix}{alias}CI}}{{{ci(summary[key]['ci95'], 2)}}}")
    correlation = snapshot.get("length_correlation")
    if correlation:
        lines.append(rf"\newcommand{{\covLengthRho}}{{{num(correlation['rho'], 2)}}}")
        lines.append(rf"\newcommand{{\covLengthRhoCI}}{{{ci(correlation['ci95'], 2)}}}")
        lines.append(rf"\newcommand{{\covLengthArms}}{{{len(correlation['arms'])}}}")
    for alias, entry in snapshot.get("offrubric_audit", {}).get("pairs", {}).items():
        # Unsuffixed macros are the primary rater's view; suffixed ones name each view.
        views = [("", entry["views"][rater_token(entry["primary"])])]
        views += [(token, row) for token, row in entry["views"].items()]
        for suffix, row in views:
            for key, name in (("actions", "Actions"), ("harm", "Harm"), ("severe", "Severe")):
                stats = row[key]
                stem = rf"\covAudit{name}"
                lines.append(
                    rf"\newcommand{{{stem}Bare{alias}{suffix}}}{{{stats['bare_per_item']:.2f}}}"
                )
                lines.append(
                    rf"\newcommand{{{stem}Prompt{alias}{suffix}}}{{{stats['prompted_per_item']:.2f}}}"
                )
                low, high = stats["ci95"]
                text = rf"\ensuremath{{{stats['difference']:+.2f}\,[{low:+.2f},\,{high:+.2f}]}}"
                lines.append(rf"\newcommand{{{stem}Diff{alias}{suffix}}}{{{text}}}")
            lines.append(rf"\newcommand{{\covAuditItems{alias}{suffix}}}{{{row['items']}}}")
        lines.append(rf"\newcommand{{\covAuditRaters{alias}}}{{{len(entry['raters'])}}}")
        if "agreement" in entry:
            agree = entry["agreement"]
            lines.append(rf"\newcommand{{\covAuditAgree{alias}}}{{{num(agree['agreement'], 2)}}}")
            lines.append(rf"\newcommand{{\covAuditKappa{alias}}}{{{num(agree['kappa'], 2)}}}")
            lines.append(
                rf"\newcommand{{\covAuditAgreeActions{alias}}}{{{int(agree['actions']):,}}}".replace(
                    ",", "{,}"
                )
            )
    trace = snapshot.get("writer_coverage_trace")
    if trace:
        for key, alias in (
            ("source_actions_median", "SourceActionsMedian"),
            ("missing_initial_median", "MissingInitialMedian"),
            ("missing_initial_mean", "MissingInitialMean"),
            ("missing_final_median", "MissingFinalMedian"),
            ("missing_final_mean", "MissingFinalMean"),
        ):
            value = trace[key]
            text = f"{value:g}" if float(value).is_integer() else f"{value:.2f}"
            lines.append(rf"\newcommand{{\cov{alias}}}{{{text}}}")
    # Every number the manuscript may quote exists, so a draft compiles while arms are running:
    # what is not measured yet reads "pending" rather than failing the build or reading as zero.
    expected = [
        f"cov{key}{alias}"
        for _, alias in ARMS.values()
        for key in (
            "F",
            "CI",
            "P",
            "R",
            "S",
            "A",
            "FDedup",
            "CIDedup",
            "FAll",
            "Off",
            "Worst",
            "Chars",
        )
    ]
    expected += [f"cov{prefix}{alias}" for *_, alias in PAIRS for prefix in ("D", "DDedup", "DAll")]
    expected += [f"covInteraction{alias}" for *_, alias in INTERACTIONS]
    expected += ["covJudgeInteraction", "covGridModels"]
    expected += [
        f"{prefix}{name}{ci}"
        for prefix in ("covGrid", "covPaperGrid", "covUncappedGrid")
        for name in GRID_NAMES.values()
        for ci in ("", "CI")
    ]
    expected += ["covPaperGridModels", "covUncappedGridModels", "covLengthRho", "covLengthRhoCI"]
    expected += ["covLengthArms"]
    expected += [
        f"covAudit{name}{kind}{alias}{suffix}"
        for _, alias in AUDIT_PAIRS.values()
        for name in ("Actions", "Harm", "Severe")
        for kind in ("Bare", "Prompt", "Diff")
        for suffix in ("", "Nemo", "Ling", "Both")
    ]
    expected += [
        f"cov{stem}{alias}"
        for _, alias in AUDIT_PAIRS.values()
        for stem in ("AuditRaters", "AuditAgree", "AuditKappa", "AuditAgreeActions")
    ]
    expected += [
        f"covAuditItems{alias}{suffix}"
        for _, alias in AUDIT_PAIRS.values()
        for suffix in ("", "Nemo", "Ling", "Both")
    ]
    defined = {line.split("{", 2)[1].lstrip("\\").rstrip("}") for line in lines[4:]}
    lines += [rf"\newcommand{{\{name}}}{{\covPending}}" for name in expected if name not in defined]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
