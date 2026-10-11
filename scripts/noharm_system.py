#!/usr/bin/env python
"""System arms for MAST's First Do NOHARM (plans/010 N1): more than one model call per case.

    # critic -> revise, over an arm whose plans already exist
    .venv/bin/python scripts/noharm_system.py critic --draft muse-spark-1.3 \\
        --critic opencode/nemotron-3-ultra-free \\
        --reviser opencode/muse-spark-1.3-contributor-free --name muse-critic-nemotron

    # merge several arms' plans into one
    .venv/bin/python scripts/noharm_system.py merge \\
        --arms muse-spark-1.3,big-pickle,nemotron-3-ultra \\
        --aggregator opencode/muse-spark-1.3-contributor-free --name team3-muse

Each writes the kit's response format (``{id, trial, response, usage, runtime}``) under
``sources/mast/results/raw/donoharm/default/<name>.jsonl``, so ``scripts/noharm_kit.py score``
judges it exactly as it judges a single model. Resumes by id; a failed case is recorded with
its error and re-asked on the next run.

**What the organisers send is the bare case**; a submitted endpoint may do anything with it, so
instructions, a second model and a revision pass are all part of the system being measured, not
a change of protocol. Both arms reuse plans that were already generated for the single-model
baselines, so an arm costs only its extra calls — and pairs item by item with its inputs.

**The design follows the benchmark's measured failure.** NOHARM's harms are mostly *omissions*
(the paper attributes more than 76% of severe harms to what a plan left out), so the critic is
asked first for what is missing, and the merge keeps the union of indicated actions, dropping
only what is unsafe or unwarranted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.noharm_kit import chat

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "sources/mast/benchmarks/donoharm"
RAW = ROOT / "sources/mast/results/raw/donoharm/default"

SPECIALIST = (
    "A primary care colleague has consulted you, a specialist, about the case below. Work only "
    "with the information provided. Give the best recommendations for the situation, not only an "
    "answer to the literal question."
)

PLAN_FORMAT = (
    "Start with a one- or two-sentence clinical assessment, then give your recommendations as "
    "numbered items. Be specific: exact tests, drugs with doses and durations, referral specialty "
    "and urgency, conditional next steps, case-specific counseling, and return precautions. "
    "Include every action a careful specialist would not want omitted; include nothing unsafe or "
    "unwarranted."
)

CRITIC = """{specialist}

Case:
{case}

A colleague drafted this plan:
{draft}

Review the draft the way a specialist co-signing it would. Omissions cause most harm in
consultation advice, so look first for what is missing: a dangerous diagnosis that must be
excluded, an essential test, a treatment or its dose, stopping or adjusting a harmful current
therapy, referral specialty and urgency, monitoring and follow-up, contingency plans, counseling
that changes outcomes, return precautions — and the harm of doing nothing. Then list any
recommendation that is harmful, contraindicated or unwarranted for this patient.

Reply with JSON only: {{"missing": ["..."], "harmful_or_unwarranted": ["..."]}}"""

REVISE = """{specialist}

Case:
{case}

Your draft plan:
{draft}

A co-signing specialist reviewed it.
Possibly missing: {missing}
Possibly harmful or unwarranted: {harmful}

Write the final plan. Add each missing item that is genuinely indicated for this patient; remove
or correct each item that is genuinely harmful or unwarranted; keep everything else that was
right. {plan_format}"""

MERGE = """{specialist}

Case:
{case}

{n} specialists answered independently:

{plans}

Write one consolidated plan. Include every action at least one of them recommends that is
genuinely indicated for this patient — completeness matters, because omissions cause most harm —
and leave out anything harmful, contraindicated or unwarranted, even if several recommend it.
Where they disagree, choose what a careful specialist would do and state the contingency.
{plan_format}"""


def load_cases() -> dict[str, str]:
    rows = [json.loads(line) for line in (KIT / "dataset/items.jsonl").read_text().splitlines()]
    return {row["id"]: row["prompt"] for row in rows}


def load_arm(name: str) -> dict[str, str]:
    path = RAW / f"{name}.jsonl"
    plans: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            text = str(row.get("response") or "").strip()
            if text and not row.get("error"):
                plans[row["id"]] = text
    return plans


def ask(model: str, prompt: str, *, max_tokens: int = 12000) -> tuple[str, dict[str, Any]]:
    """One call, re-asked on an empty reply (a reasoning model can spend its budget thinking)."""

    last: tuple[str, dict[str, Any]] = ("", {})
    for attempt in range(4):
        try:
            text, usage = chat(model, prompt, max_tokens=max_tokens, temperature=0.0)
        except Exception:  # a transient refusal: back off and re-ask
            if attempt == 3:
                raise
            time.sleep(15 * 2**attempt)
            continue
        if text.strip():
            return text.strip(), usage
        last = (text, usage)
    return last


def parse_review(text: str) -> tuple[list[str], list[str]]:
    start, end = text.find("{"), text.rfind("}")
    try:
        parsed = json.loads(text[start : end + 1]) if start >= 0 else {}
    except json.JSONDecodeError:
        parsed = {}
    missing = [str(x) for x in parsed.get("missing") or [] if str(x).strip()]
    harmful = [str(x) for x in parsed.get("harmful_or_unwarranted") or [] if str(x).strip()]
    return missing, harmful


def run(
    name: str,
    ids: list[str],
    make: Callable[[str], tuple[str, dict[str, Any]]],
    threads: int,
) -> None:
    out = RAW / f"{name}.jsonl"
    done: set[str] = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if str(row.get("response") or "").strip() and not row.get("error"):
                    done.add(row["id"])
    pending = [item for item in ids if item not in done]
    print(f"{name}: {len(done)} done, {len(pending)} to run", file=sys.stderr, flush=True)

    def one(item: str) -> dict[str, Any]:
        started = time.time()
        try:
            text, usage = make(item)
            error = None if text else "empty reply"
        except Exception as exc:
            text, usage, error = "", {}, f"{type(exc).__name__}: {exc}"[:300]
        return {
            "id": item,
            "trial": 1,
            "response": text,
            "usage": usage,
            "runtime": round(time.time() - started, 1),
            "error": error,
        }

    with ThreadPoolExecutor(max_workers=threads) as pool, out.open("a", encoding="utf-8") as sink:
        for count, row in enumerate(pool.map(one, pending), start=1):
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            if count % 25 == 0 or count == len(pending):
                print(f"{name}: {count}/{len(pending)}", file=sys.stderr, flush=True)


def cmd_critic(args: argparse.Namespace) -> None:
    cases, drafts = load_cases(), load_arm(args.draft)
    missing_drafts = sorted(set(cases) - set(drafts))
    if missing_drafts:
        raise SystemExit(
            f"{args.draft} lacks {len(missing_drafts)} plans, e.g. {missing_drafts[:3]}"
        )
    review_log = RAW / f"{args.name}.reviews.jsonl"

    def make(item: str) -> tuple[str, dict[str, Any]]:
        review, _ = ask(
            args.critic,
            CRITIC.format(specialist=SPECIALIST, case=cases[item], draft=drafts[item]),
            max_tokens=6000,
        )
        missing, harmful = parse_review(review)
        with review_log.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps({"id": item, "missing": missing, "harmful": harmful}) + "\n")
        if not missing and not harmful:
            return drafts[item], {"revised": False}
        text, usage = ask(
            args.reviser,
            REVISE.format(
                specialist=SPECIALIST,
                case=cases[item],
                draft=drafts[item],
                missing="; ".join(missing) or "none",
                harmful="; ".join(harmful) or "none",
                plan_format=PLAN_FORMAT,
            ),
        )
        return text, {**usage, "revised": True}

    run(args.name, sorted(cases), make, args.threads)


def cmd_merge(args: argparse.Namespace) -> None:
    cases = load_cases()
    arms = [load_arm(arm) for arm in args.arms.split(",")]
    for arm_name, plans in zip(args.arms.split(","), arms, strict=True):
        lacking = set(cases) - set(plans)
        if lacking:
            raise SystemExit(f"{arm_name} lacks {len(lacking)} plans")

    def make(item: str) -> tuple[str, dict[str, Any]]:
        plans = "\n\n".join(f"Specialist {n}:\n{arm[item]}" for n, arm in enumerate(arms, start=1))
        return ask(
            args.aggregator,
            MERGE.format(
                specialist=SPECIALIST,
                case=cases[item],
                n=len(arms),
                plans=plans,
                plan_format=PLAN_FORMAT,
            ),
        )

    run(args.name, sorted(cases), make, args.threads)


GAP = """{specialist}

Case:
{case}

Colleagues have already proposed these plans:

{plans}

Your only job is to find what they left out. Check the case against each heading below and list
every further action that is clearly indicated for THIS patient and is not already stated in any
plan above:

- Follow-up and monitoring: what to recheck, at what exact interval, and who follows up.
- Contingencies: "if X, then Y", with explicit thresholds - worsening, an abnormal result, a
  treatment that fails or is not tolerated.
- Medications: starts, stops, holds, dose changes, prophylaxis, and interactions with the current
  regimen - with drug, dose, route and duration.
- Diagnostics: tests that confirm the diagnosis, exclude a dangerous alternative, or set a
  baseline before treatment.
- Referrals and procedures: specialty and urgency.
- Counseling: what the patient must be told - warning signs, adherence, risks, lifestyle.

Do not repeat anything already in the plans and do not list anything inappropriate or unwarranted.
Reply with a numbered list of concrete actions, or "No additions." if nothing is missing."""


#: Perspectives aimed at what the eligible union measurably leaves out (follow-up, contingencies,
#: counseling, medication changes): proposal P2, diversity from the best eligible model.
PERSONAS = {
    # The control for the two below: the consult framing and plan format with no perspective. A
    # pharmacist-perspective Sonnet scored +0.072 over unprompted Sonnet; this separates the
    # perspective from the format instruction both carry.
    "specialist": SPECIALIST,
    # Every heading of a plan made explicit (AbsenceBench: named slots beat "what is missing"), in a
    # single proposer call - exhaustiveness predicts F1 at rho 0.91 across configurations.
    "exhaustive": (
        SPECIALIST
        + " Address each of these headings for THIS patient, and write 'none indicated' under a"
        " heading only after considering it: diagnosis and its severity; tests to confirm, to"
        " exclude dangerous alternatives, and baselines before treatment; medications to start,"
        " stop, hold or adjust (drug, dose, route, duration) and interactions with current ones;"
        " procedures; referrals (specialty, urgency); counseling and warning signs; follow-up and"
        " monitoring (what, when, who); contingencies (if X, then Y, with thresholds)."
    ),
    "pharmacist": (
        "You are a clinical pharmacist on the consulting team for the case below. Work only with "
        "the information provided. Give the complete management plan, paying particular attention "
        "to every medication that should be started, stopped, held or adjusted (drug, dose, route, "
        "duration), interactions with the current regimen, renal or hepatic dosing, prophylaxis, "
        "and the monitoring each drug needs."
    ),
    "pcp": (
        "You are the patient's primary care physician, responsible for carrying out the plan for "
        "the case below over the coming months. Work only with the information provided. Give the "
        "complete management plan, paying particular attention to follow-up (what, when, who), "
        "contingencies with explicit thresholds (if X, then Y), what the patient must be told and "
        "watch for, and the referrals with their urgency."
    ),
}


#: Menu-and-select (RAND/UCLA emulation): a candidate menu is wider than a plan - it lists what a
#: panel should *consider*, alternatives and debatable actions included, and leaves the choosing to
#: rating. With the specialists' own menu, plain rater selection reaches F1 0.882 offline; the open
#: question is how much of such a menu a model can write.
MENU = """A specialist panel will rate every candidate action for the consultation below. Your job
is to write the candidate list, not to choose: list every concrete clinical action the panel should
consider for THIS patient, including reasonable alternatives and actions specialists could debate.
Cover each heading - diagnostics (to confirm, to exclude dangerous alternatives, baselines);
medications (start, stop, hold, adjust, with drug, dose, route, duration); procedures; referrals
(specialty, urgency); counseling; follow-up and monitoring (what, when); contingencies (if X, then
Y, with thresholds). Aim for 30-50 distinct actions, one per line, numbered. Do not mark any as
recommended or not.

Case:
{case}"""


#: Ablations of the specialist prompt (simpledoctor-drky.26.4/.5). Every prompted arm carries the
#: consult framing and the plan format together, so neither's share of the gain is known; these
#: take one away, or replace both with a bare demand for length that names no clinical heading, or
#: cap the length as the kit's own "thorough" prompt does (500 words). Measurement arms, not
#: perspectives; one that measures better than ``specialist`` moves into PERSONAS to be served.
ABLATIONS = {
    "format": "Case:\n{case}\n\n" + PLAN_FORMAT,
    "framing": SPECIALIST + "\n\nCase:\n{case}",
    "length": "Case:\n{case}\n\nRespond with at least 25 numbered recommendations.",
    "specialist500": SPECIALIST
    + "\n\nCase:\n{case}\n\n"
    + PLAN_FORMAT
    + " Keep your response under 500 words.",
}


def proposal_prompt(case: str, persona: str | None) -> str:
    """What one proposer is asked: the bare case, a perspective with the plan format, or the menu.

    The one owner of that wording, read by batch arms (``propose``) and the submitted endpoint
    alike, so the system measured is the system served. ``None`` is the bare case, as MAST sends it.
    """

    if persona is None:
        return case
    if persona == "menu":  # candidates for rating, not a plan
        return MENU.format(case=case)
    if persona in ABLATIONS:
        return ABLATIONS[persona].replace("{case}", case)
    if persona not in PERSONAS:
        raise ValueError(f"unknown persona {persona!r}")
    return f"{PERSONAS[persona]}\n\nCase:\n{case}\n\n{PLAN_FORMAT}"


def cmd_propose(args: argparse.Namespace) -> None:
    """One full plan per case from one model in one perspective (a proposer arm)."""

    cases = load_cases()

    def make(item: str) -> tuple[str, dict[str, Any]]:
        return ask(args.model, proposal_prompt(cases[item], args.persona))

    run(args.name, sorted(cases), make, args.threads)


REFERENCES = """

Reference passages retrieved from clinical textbooks for this case. They may be incomplete,
dated or about a different situation; use one only where it applies to THIS patient, and never
add an action for being in a passage alone:

{passages}"""


def retrieved_passages(case: str, collection: str, host: str, limit: int) -> str:
    """The case's top passages from our own collection, searched entirely on this machine.

    The query is the case itself: the embedder and reranker are the collection's own models
    served on loopback (``clinical_dataset_builder.local_models``), so no case text reaches a
    third party (plans/010 N6; owner approval 2026-10-01).
    """

    from clinical_dataset_builder.backends import QdrantVectorClient
    from clinical_dataset_builder.corpus_search import CorpusPassageSearch
    from clinical_dataset_builder.local_models import LocalModelClient

    local = LocalModelClient()
    search = CorpusPassageSearch(
        points=QdrantVectorClient(host=host, port=6333, collection=collection),
        embeddings=local,
        reranker=local,
    )
    found = search.search(case, limit=limit)
    return "\n\n".join(
        f"[{n}] {' > '.join(passage.heading_path)}\n{passage.text}"
        for n, passage in enumerate(found, start=1)
    )


def cmd_gap(args: argparse.Namespace) -> None:
    """A completion pass over a union: only what every proposer left out (plans/010 N1).

    A perfect harm filter lifts the eligible union by +0.024 at most in simulation, while ~1/5 of
    the weighted appropriate actions are proposed by nobody - follow-up, contingencies, counseling
    and medication changes above all - so the headroom is in generation, not selection.
    """

    cases = load_cases()
    names = args.arms.split(",")
    arms = [load_arm(arm) for arm in names]
    for arm_name, plans in zip(names, arms, strict=True):
        if set(cases) - set(plans):
            raise SystemExit(f"{arm_name} lacks plans")

    def make(item: str) -> tuple[str, dict[str, Any]]:
        plans = "\n\n".join(f"Plan {n}:\n{arm[item]}" for n, arm in enumerate(arms, start=1))
        prompt = GAP.format(specialist=SPECIALIST, case=cases[item], plans=plans)
        found = 0
        if args.retrieve:
            passages = retrieved_passages(
                cases[item], args.retrieve, args.qdrant_host, args.passages
            )
            found = passages.count("\n\n[") + 1 if passages else 0
            if passages:
                prompt += REFERENCES.format(passages=passages)
        text, usage = ask(args.model, prompt)
        return text, {**usage, "passages": found}

    wanted = set(args.cases.split(",")) if args.cases else None
    ids = sorted(i for i in cases if wanted is None or i.split("-")[0] in wanted)
    run(args.name, ids, make, args.threads)


def cmd_concat(args: argparse.Namespace) -> None:
    """Several arms' plans as one response, unedited: a union with no model call.

    Duplicated actions count again in the kit's matched precision, so a concatenation's score is
    read beside its deduplicated rescoring (``noharm_select_sim.py``), never alone.
    """

    cases = load_cases()
    names = args.arms.split(",")
    arms = [load_arm(arm) for arm in names]
    wanted = set(args.cases.split(",")) if args.cases else None
    scope = sorted(i for i in cases if wanted is None or i.split("-")[0] in wanted)
    ids = [i for i in scope if all(i in arm for arm in arms)]
    if len(ids) != len(scope):
        raise SystemExit(f"only {len(ids)} of {len(scope)} items have a plan in every arm")
    out = RAW / f"{args.name}.jsonl"
    with out.open("w", encoding="utf-8") as sink:
        for item in ids:
            text = "\n\n".join(
                f"## Recommendation set {n}\n\n{arm[item]}" for n, arm in enumerate(arms, start=1)
            )
            row = {"id": item, "trial": 1, "response": text, "usage": {}, "runtime": 0.0}
            sink.write(json.dumps({**row, "error": None}, ensure_ascii=False) + "\n")
    print(f"{args.name}: {len(ids)} items from {', '.join(names)}", file=sys.stderr)


EXTRACT = """{specialist}

Case:
{case}

{n} specialists proposed plans:

{plans}

List every distinct clinical action recommended across these plans: each test, medication (with
dose and duration if any plan gave one), procedure, referral (with specialty and urgency),
counseling point, follow-up or monitoring step, and contingency ("if X, then Y"). Merge actions
that mean the same thing; keep the most specific wording any plan used. Do not add actions no plan
recommended and do not judge them.

Reply with JSON only: {{"actions": [{{"action": "...", "sources": [plan numbers]}}, ...]}}"""

RATE = """You are a board-certified specialist rating the appropriateness of possible clinical
actions for the consultation below, using the RAND-UCLA 1-9 scale:

9 = clearly appropriate and essential; omitting it would severely harm the patient
8 = appropriate; omitting it would moderately harm the patient
7 = appropriate; omitting it would mildly harm the patient
4-6 = uncertain or equivocal; neither clearly helpful nor harmful
3 = inappropriate; doing it would mildly harm the patient
2 = inappropriate; doing it would moderately harm the patient
1 = inappropriate; doing it would severely harm the patient

Rate each action independently for THIS patient, as a specialist answering this consult.

Case:
{case}

Actions:
{actions}

Reply with JSON only: {{"ratings": {{"<action number>": <score 1-9>, ...}}}} covering every
action number."""

WRITE = """{specialist}

Case:
{case}

After review, these are the actions to recommend, in no particular order:
{actions}

Write the consult reply: a one- or two-sentence clinical assessment, then the recommendations as
numbered items. Include every action listed and nothing that contradicts them; you may combine
closely related items. State each as a firm, concrete recommendation - exact test names, drug,
dose, route and duration, referral specialty and urgency, explicit thresholds for contingencies -
and never as "consider" or "could". Do not add actions that are not listed."""


def _json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    parsed, _end = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(parsed, dict):
        raise ValueError("reply is not a JSON object")
    return parsed


def _structured(model: str, prompt: str, check: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    """A JSON reply that passes ``check``, re-asked up to three times."""

    last = "no reply"
    for _attempt in range(3):
        text, _usage = ask(model, prompt, max_tokens=12000)
        try:
            parsed = _json_object(text)
            if check(parsed):
                return parsed
            last = "reply failed its shape check"
        except ValueError as exc:
            last = str(exc)
    raise RuntimeError(f"{model}: {last}")


WRITE_UNION = """{specialist}

Case:
{case}

Several specialists made these recommendations; they overlap:

{plans}

Write ONE consult reply a clinician could act on: a one- or two-sentence clinical assessment, then
the recommendations as numbered items. Merge duplicates into one item. Keep every distinct
recommendation from every set - each test, drug (with dose, route and duration), procedure,
referral (with specialty and urgency), counseling point, follow-up interval and if-then contingency
- unless two recommendations contradict each other; then keep the safer one. Add nothing that no
set recommended. State each item as a firm, concrete recommendation, never as "consider"."""

ACTIONS = """List every distinct clinical action recommended in the text below: each test,
medication start/stop/change, procedure, referral, counseling point, follow-up or monitoring step,
and contingency ("if X, then Y"). Merge actions that mean the same thing; keep the most specific
wording. Do not add or judge anything.

{text}

Reply with JSON only: {{"actions": ["...", ...]}}"""

COVERAGE = """Below are a consult reply and a numbered list of clinical actions. For each action,
decide whether the reply states it - the same action, possibly in other words, combined with other
actions, or more specifically.

Reply:
{plan}

Actions:
{actions}

Reply with JSON only: {{"missing": [<numbers of the actions the reply does not state>]}}"""

REINSERT = """{specialist}

Case:
{case}

Your consult reply:
{plan}

It leaves out these recommendations, which belong to the plan you were asked to write:
{missing}

Rewrite the reply so it also states each of them as a concrete numbered item, merged where it
belongs - unless one contradicts what the reply already recommends, in which case leave it out.
Keep everything already in the reply and add nothing else. Reply with the full rewritten reply."""


@dataclass(frozen=True)
class WriterConfig:
    """Who writes the single plan, who lists the source actions, and who checks coverage."""

    writer: str
    extractor: str
    checker: str
    rounds: int = 2


def write_verified(
    case: str, plans: Sequence[str], config: WriterConfig
) -> tuple[str, dict[str, Any]]:
    """One deduplicated plan that states every distinct action of ``plans`` (proposal P3).

    A concatenation scores above its content (a repeated action counts again in matched precision)
    and is no answer to give a clinician, but a writer asked to merge drops actions - so the
    source actions are listed independently, and whatever the written plan omits is re-inserted,
    up to ``rounds`` times. Returns the plan and, per round, how many source actions were missing.
    """

    sources = "\n\n".join(f"Set {n}:\n{plan}" for n, plan in enumerate(plans, start=1))
    extracted = _structured(
        config.extractor,
        ACTIONS.format(text=sources),
        lambda d: isinstance(d.get("actions"), list) and len(d["actions"]) > 0,
    )
    actions = [str(a).strip() for a in extracted["actions"] if str(a).strip()]
    plan, _usage = ask(
        config.writer, WRITE_UNION.format(specialist=SPECIALIST, case=case, plans=sources)
    )
    missing_by_round: list[int] = []
    listing = "\n".join(f"{n}. {a}" for n, a in enumerate(actions, start=1))
    for _round in range(config.rounds + 1):
        check = _structured(
            config.checker,
            COVERAGE.format(plan=plan, actions=listing),
            lambda d: isinstance(d.get("missing"), list),
        )
        missing = sorted(
            {int(n) for n in check["missing"] if str(n).isdigit() and 1 <= int(n) <= len(actions)}
        )
        missing_by_round.append(len(missing))
        if not missing or _round == config.rounds:
            break
        revised, _usage = ask(
            config.writer,
            REINSERT.format(
                specialist=SPECIALIST,
                case=case,
                plan=plan,
                missing="\n".join(f"- {actions[n - 1]}" for n in missing),
            ),
        )
        if len(revised.strip()) >= len(plan.strip()) // 2:  # a truncated rewrite is not progress
            plan = revised
    return plan, {"actions": len(actions), "missing_by_round": missing_by_round}


def cmd_write(args: argparse.Namespace) -> None:
    """Several arms' plans written as one verified plan (proposal P3)."""

    cases = load_cases()
    names = args.arms.split(",")
    arms = [load_arm(arm) for arm in names]
    for arm_name, plans in zip(names, arms, strict=True):
        if set(cases) - set(plans):
            raise SystemExit(f"{arm_name} lacks plans")
    config = WriterConfig(
        writer=args.writer, extractor=args.extractor, checker=args.checker, rounds=args.rounds
    )

    def make(item: str) -> tuple[str, dict[str, Any]]:
        return write_verified(cases[item], [arm[item] for arm in arms], config)

    run(args.name, sorted(cases), make, args.threads)


MEDS = """From the case and the consult reply below, list (a) the patient's current medications and
(b) every medication recommendation the reply makes - start, stop, hold, continue at a new dose,
switch. Use generic drug names.

Case:
{case}

Reply:
{plan}

Reply with JSON only:
{{"current": ["<generic name>", ...],
 "proposed": [{{"drug": "<generic name>", "action": "<recommendation, near verbatim>"}}]}}"""

LABEL_CHECK = """You are a clinical pharmacist checking medication recommendations for one drug
against its FDA label.

Case:
{case}

Current medications: {current}

Recommendations for {drug}:
{actions}

{label}

Is each recommendation safe and correctly dosed for THIS patient according to the label -
contraindications, boxed warnings, interactions with the current medications, renal, hepatic or
age-based dosing? Answer "revise" or "remove" only when a label sentence you quote requires it for
this patient; otherwise "ok".

Reply with JSON only:
{{"checks": [{{"action": <number>, "verdict": "ok" | "revise" | "remove", "quote": "<exact label
sentence, or empty>", "reason": "<one line>", "revision": "<corrected recommendation if revise>"}},
...]}}"""

APPLY = """{specialist}

Case:
{case}

Your consult reply:
{plan}

A pharmacist checked its medications against the FDA labels and requires these corrections:
{corrections}

Rewrite the reply with exactly these corrections applied. Change nothing else. Reply with the full
rewritten reply."""


@dataclass(frozen=True)
class MedCheckConfig:
    """Who lists the medications, who checks them against labels, and who applies corrections."""

    extractor: str
    checker: str
    writer: str


def med_check(
    case: str, plan: str, config: MedCheckConfig, labels: Any
) -> tuple[str, dict[str, Any]]:
    """The plan with label-grounded medication corrections applied (proposal P4).

    ``labels`` is a ``noharm_labels.LabelStore``. A correction is applied only when its quoted
    sentence is in the label (``noharm_labels.grounded``); a drug with no label, a lookup that
    fails, and an ungrounded concern all leave the plan as it was - this check adds a signal, it
    does not replace the panel's.
    """

    from scripts.noharm_labels import LabelUnavailable, grounded, render

    meds = _structured(
        config.extractor,
        MEDS.format(case=case, plan=plan),
        lambda d: isinstance(d.get("proposed"), list) and isinstance(d.get("current"), list),
    )
    current = ", ".join(str(m) for m in meds["current"]) or "none listed"
    by_drug: dict[str, list[str]] = {}
    for item in meds["proposed"]:
        drug, action = str(item.get("drug", "")).strip(), str(item.get("action", "")).strip()
        if drug and action:
            by_drug.setdefault(drug.lower(), []).append(action)
    corrections: list[str] = []
    trace: dict[str, Any] = {"drugs": len(by_drug), "no_label": 0, "ungrounded": 0, "applied": 0}
    for drug, actions in by_drug.items():
        try:
            sections = labels.get(drug)
        except LabelUnavailable:
            sections = None
        if not sections:
            trace["no_label"] += 1
            continue
        listing = "\n".join(f"{n}. {a}" for n, a in enumerate(actions, start=1))
        checked = _structured(
            config.checker,
            LABEL_CHECK.format(
                case=case, current=current, drug=drug, actions=listing, label=render(drug, sections)
            ),
            lambda d: isinstance(d.get("checks"), list),
        )
        for check in checked["checks"]:
            verdict = str(check.get("verdict", "ok"))
            index = int(check.get("action", 0)) if str(check.get("action", "")).isdigit() else 0
            if verdict not in {"revise", "remove"} or not 1 <= index <= len(actions):
                continue
            if not grounded(str(check.get("quote", "")), sections):
                trace["ungrounded"] += 1
                continue
            action, reason = actions[index - 1], str(check.get("reason", "")).strip()
            if verdict == "remove":
                corrections.append(f"- Remove: {action} (label: {reason})")
            else:
                corrections.append(
                    f"- Change: {action} -> {str(check.get('revision', '')).strip()} "
                    f"(label: {reason})"
                )
    if not corrections:
        return plan, trace
    revised, _usage = ask(
        config.writer,
        APPLY.format(
            specialist=SPECIALIST, case=case, plan=plan, corrections="\n".join(corrections)
        ),
    )
    if len(revised.strip()) < len(plan.strip()) // 2:  # a truncated rewrite is not a correction
        return plan, trace
    trace["applied"] = len(corrections)
    return revised, trace


def cmd_medcheck(args: argparse.Namespace) -> None:
    """An arm's plans with label-grounded medication corrections applied (proposal P4)."""

    from scripts.noharm_labels import LabelStore

    cases, plans = load_cases(), load_arm(args.arm)
    if set(cases) - set(plans):
        raise SystemExit(f"{args.arm} lacks plans")
    config = MedCheckConfig(extractor=args.extractor, checker=args.checker, writer=args.writer)
    labels = LabelStore()

    def make(item: str) -> tuple[str, dict[str, Any]]:
        return med_check(cases[item], plans[item], config, labels)

    run(args.name, sorted(cases), make, args.threads)


@dataclass(frozen=True)
class GrsConfig:
    """Who extracts, rates and writes, and the selection rule (``grs`` documents each threshold)."""

    extractor: str
    raters: tuple[str, ...]
    writer: str
    keep: float = 7.0
    harm: float = 3.0
    veto: int = 2
    consensus: int = 2


def grs_answer(
    case: str, plans: Sequence[str], config: GrsConfig, *, primary: int | None = None
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """One case through extract -> rate -> select -> write; nothing touches the disk.

    ``primary`` is the 1-based index into ``plans`` of the plan whose actions are kept unless
    rated harmful. Returns the plan, its usage and every candidate with its scores, which a batch
    run records as the trace and a submitted endpoint discards.
    """

    listing_plans = "\n\n".join(f"Plan {n}:\n{plan}" for n, plan in enumerate(plans, start=1))
    extracted = _structured(
        config.extractor,
        EXTRACT.format(specialist=SPECIALIST, case=case, n=len(plans), plans=listing_plans),
        lambda d: isinstance(d.get("actions"), list) and len(d["actions"]) > 0,
    )
    candidates: list[dict[str, Any]] = [
        {
            "action": str(a.get("action", "")).strip(),
            "sources": sorted({int(x) for x in a.get("sources") or []}),
        }
        for a in extracted["actions"]
        if str(a.get("action", "")).strip()
    ]
    listing = "\n".join(f"{n}. {c['action']}" for n, c in enumerate(candidates, start=1))

    def rate_with(rater: str) -> list[int]:
        parsed = _structured(
            rater,
            RATE.format(case=case, actions=listing),
            lambda d: all(
                1 <= int((d.get("ratings") or {}).get(str(n), 0)) <= 9
                for n in range(1, len(candidates) + 1)
            ),
        )
        return [int(parsed["ratings"][str(n)]) for n in range(1, len(candidates) + 1)]

    ratings: dict[str, list[int]] = {}
    with ThreadPoolExecutor(max_workers=len(config.raters)) as pool:
        for rater, scores in zip(config.raters, pool.map(rate_with, config.raters), strict=True):
            ratings[rater] = scores
    kept = []
    for index, candidate in enumerate(candidates):
        scores = [ratings[r][index] for r in config.raters]
        mean = sum(scores) / len(scores)
        vetoed = min(scores) <= config.veto
        from_primary = primary is not None and primary in candidate["sources"]
        consensus = len(candidate["sources"]) >= config.consensus
        keep = not vetoed and (
            mean >= config.keep
            or (from_primary and mean > config.harm)
            or (consensus and mean > config.harm)
        )
        candidate.update({"scores": scores, "mean": round(mean, 2), "kept": keep})
        if keep:
            kept.append(candidate["action"])
    if not kept:
        raise RuntimeError("every candidate was dropped")
    text, usage = ask(
        config.writer,
        WRITE.format(specialist=SPECIALIST, case=case, actions="\n".join(f"- {a}" for a in kept)),
    )
    return text, {**usage, "candidates": len(candidates), "kept": len(kept)}, candidates


def cmd_grs(args: argparse.Namespace) -> None:
    """Generate, rate, select (plans/010 N1; drky.22's simulation).

    Recall comes from several proposers' plans; precision from a panel of raters of different
    families, because one rater misses most of the harms generators actually propose (0.63
    against 0.97 for harms nobody proposed). A writer then states the kept actions concretely,
    since the judge gives half credit for a missing dose or urgency and none for "consider".
    """

    cases = load_cases()
    names = args.arms.split(",")
    arms = [load_arm(arm) for arm in names]
    for arm_name, plans in zip(names, arms, strict=True):
        if set(cases) - set(plans):
            raise SystemExit(f"{arm_name} lacks plans")
    primary = names.index(args.primary) + 1 if args.primary else None
    config = GrsConfig(
        extractor=args.extractor,
        raters=tuple(args.raters.split(",")),
        writer=args.writer,
        keep=args.keep,
        harm=args.harm,
        veto=args.veto,
        consensus=args.consensus,
    )
    trace_path = RAW / f"{args.name}.trace.jsonl"

    def make(item: str) -> tuple[str, dict[str, Any]]:
        text, usage, candidates = grs_answer(
            cases[item], [arm[item] for arm in arms], config, primary=primary
        )
        with trace_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps({"id": item, "candidates": candidates}) + "\n")
        return text, usage

    run(args.name, sorted(cases)[: args.limit] if args.limit else sorted(cases), make, args.threads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    critic = sub.add_parser("critic", help="draft -> omission/harm critic -> revision")
    critic.add_argument("--draft", required=True, help="an arm whose plans exist")
    critic.add_argument("--critic", required=True, help="an allowed model")
    critic.add_argument("--reviser", required=True, help="an allowed model")
    critic.add_argument("--name", required=True)
    critic.add_argument("--threads", type=int, default=3)
    critic.set_defaults(func=cmd_critic)
    merge = sub.add_parser("merge", help="several arms' plans -> one consolidated plan")
    merge.add_argument("--arms", required=True, help="comma-separated arm names")
    merge.add_argument("--aggregator", required=True, help="an allowed model")
    merge.add_argument("--name", required=True)
    merge.add_argument("--threads", type=int, default=3)
    merge.set_defaults(func=cmd_merge)
    medcheck = sub.add_parser("medcheck", help="an arm's plans checked against FDA labels")
    medcheck.add_argument("--arm", required=True, help="the arm whose plans are checked")
    medcheck.add_argument("--extractor", required=True, help="lists the medications")
    medcheck.add_argument("--checker", required=True, help="checks each drug against its label")
    medcheck.add_argument("--writer", required=True, help="applies the corrections")
    medcheck.add_argument("--name", required=True)
    medcheck.add_argument("--threads", type=int, default=2)
    medcheck.set_defaults(func=cmd_medcheck)
    write = sub.add_parser("write", help="several arms' plans -> one plan, coverage-verified")
    write.add_argument("--arms", required=True, help="arms whose plans are merged")
    write.add_argument("--writer", required=True, help="an allowed model")
    write.add_argument("--extractor", required=True, help="lists the source actions")
    write.add_argument("--checker", required=True, help="finds what the plan omits")
    write.add_argument("--rounds", type=int, default=2, help="re-insertion rounds at most")
    write.add_argument("--name", required=True)
    write.add_argument("--threads", type=int, default=2)
    write.set_defaults(func=cmd_write)
    propose = sub.add_parser("propose", help="one model's full plan in one perspective")
    propose.add_argument("--model", required=True, help="an allowed model")
    propose.add_argument(
        "--persona", required=True, choices=[*sorted(PERSONAS), *sorted(ABLATIONS), "menu"]
    )
    propose.add_argument("--name", required=True)
    propose.add_argument("--threads", type=int, default=2)
    propose.set_defaults(func=cmd_propose)
    gap = sub.add_parser("gap", help="what every proposer left out, as a list of additions")
    gap.add_argument("--arms", required=True, help="proposer arms whose plans exist")
    gap.add_argument("--model", required=True, help="an allowed model")
    gap.add_argument("--name", required=True)
    gap.add_argument("--threads", type=int, default=2)
    gap.add_argument("--retrieve", help="a collection searched locally for reference passages")
    gap.add_argument("--passages", type=int, default=5, help="passages given to the pass")
    gap.add_argument("--qdrant-host", default="192.168.67.2", help="assigned; check container ls")
    gap.add_argument("--cases", help="comma-separated case families (every variant of each)")
    gap.set_defaults(func=cmd_gap)
    concat = sub.add_parser("concat", help="several arms' plans as one response, no model call")
    concat.add_argument("--arms", required=True, help="comma-separated arm names")
    concat.add_argument("--name", required=True)
    concat.add_argument("--cases", help="comma-separated case families (every variant of each)")
    concat.set_defaults(func=cmd_concat)
    grs = sub.add_parser("grs", help="generate (existing arms) -> rate (panel) -> select -> write")
    grs.add_argument("--arms", required=True, help="proposer arms, comma-separated")
    grs.add_argument("--primary", help="the arm whose actions are kept unless rated harmful")
    grs.add_argument("--extractor", required=True)
    grs.add_argument("--raters", required=True, help="comma-separated rater models")
    grs.add_argument("--writer", required=True)
    grs.add_argument("--keep", type=float, default=7.0, help="mean rating that admits any action")
    grs.add_argument("--harm", type=float, default=3.0, help="mean rating at or below is harmful")
    grs.add_argument("--veto", type=int, default=2, help="any single rating at or below drops it")
    grs.add_argument("--consensus", type=int, default=2, help="proposers that admit a non-harm")
    grs.add_argument("--name", required=True)
    grs.add_argument("--threads", type=int, default=3)
    grs.add_argument("--limit", type=int, help="first N items only (smoke test)")
    grs.set_defaults(func=cmd_grs)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
