#!/usr/bin/env python
"""Run MAST's First Do NOHARM kit against the owner's allowed models (simpledoctor-drky.21).

    # generate plans for the 330 open-subset items with one allowed model (unprompted protocol)
    sources/mast/benchmarks/donoharm/.venv/bin/python scripts/noharm_kit.py run \\
        --model opencode/muse-spark-1.3-contributor-free --name muse-spark-1.3

    # judge and score them with the kit's own match + review pipeline and metrics
    sources/mast/benchmarks/donoharm/.venv/bin/python scripts/noharm_kit.py score \\
        --name muse-spark-1.3 --judge opencode/nemotron-3-ultra-free

Runs with the kit's own interpreter (``sources/mast/benchmarks/donoharm/.venv``), because the kit
pins its own litellm, pandas and numpy; nothing here imports this repository.

**Only the judge's model changes.** The kit judges with two Gemini models through Google's SDK,
which the owner's rule excludes (``bd recall only-opencode-free-models``). This wrapper replaces
the one function the kit calls for a judgement — prompt and JSON schema in, JSON text out — with a
call to an allowed model on ``127.0.0.1:8317``, validating the reply against the same schema
locally. The prompts, the match and review stages, the overrides and the metrics are the kit's,
unmodified. A score produced here is therefore the kit's metric under a *different judge*: it is
for comparing our arms with each other on the open subset, never for comparison with the
leaderboard, whose number comes from the organisers' own run of the held-out cases.

The open subset is CC BY 4.0 and public, so an allowed free model may see it. The organisers'
held-out cases are another matter (``plans/010`` N2).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import random
import re
import runpy
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "sources/mast/benchmarks/donoharm"
ENDPOINT = "http://127.0.0.1:8317/v1"


def endpoint_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("EVAL_8317_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("EVAL_8317_API_KEY is not in .env")


class PermanentJudgeError(RuntimeError):
    """A request the endpoint refused outright: re-asking it only repeats the refusal."""


#: Requests each of the proxy's accounts may carry at once, across every process on this machine.
#: Each account rate-limits on its own; past these, requests come back 429/503 and the retries cost
#: more than the parallelism buys (measured 2026-10-01: a generate-rate-select item took 3-10
#: minutes under contention against ~60 s of model time). ``NOHARM_SLOTS_<ACCOUNT>`` overrides.
_ACCOUNT_SLOTS = {"anthropic": 6, "codex": 3, "xai": 2, "opencode": 2, "openrouter": 4}
_SLOT_DIR = Path(tempfile.gettempdir()) / "noharm_account_slots"
_COOLDOWN_SECONDS = 30.0

#: A model addressed ``openrouter/<id>`` is asked on OpenRouter directly, not through the proxy, and
#: only if OpenRouter lists it at zero price (owner, 2026-10-10: further runs on free models only).
#: Free routes may log or train on prompts, so this carries the public open subset only. Each
#: configured OpenRouter account has its own daily free-model allowance (1000 requests funded, 50
#: unfunded); a spent account is skipped until the process ends.
OPENROUTER_PREFIX = "openrouter/"
OPENROUTER_CATALOG = "https://openrouter.ai/api/v1/models"
_SPENT_OPENROUTER: set[str] = set()
_FREE_ON_OPENROUTER: dict[str, bool] = {}


def account_of(model: str) -> str:
    if model.startswith(OPENROUTER_PREFIX):
        return "openrouter"
    if model.startswith("claude") or model in {"sonnet", "haiku", "opus"}:
        return "anthropic"
    if model.startswith("gpt-"):
        return "codex"
    if model.startswith("grok"):
        return "xai"
    return "opencode"


@contextlib.contextmanager
def account_slot(model: str) -> Iterator[None]:
    """Hold one of the account's slots for the length of a request; wait out a 429 cooldown.

    A waiter that finds every slot taken *blocks* on one rather than polling: polling lost every
    race to a busy process re-taking the slot it had just released, and starved generate-rate-
    select's rater for an hour (2026-10-01) while a merge held the same account.
    """

    account = account_of(model)
    slots = int(os.environ.get(f"NOHARM_SLOTS_{account.upper()}", _ACCOUNT_SLOTS[account]))
    _SLOT_DIR.mkdir(exist_ok=True)
    cooldown = _SLOT_DIR / f"{account}.cooldown"
    while True:
        try:
            until = float(cooldown.read_text())
        except (OSError, ValueError):
            until = 0.0
        if until > time.time():
            time.sleep(min(5.0, until - time.time()))
            continue
        handle = None
        for index in range(slots):
            candidate = (_SLOT_DIR / f"{account}-{index}.lock").open("w")
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                candidate.close()
                continue
            handle = candidate
            break
        if handle is None:
            handle = (_SLOT_DIR / f"{account}-{random.randrange(slots)}.lock").open("w")
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                waited_into_cooldown = float(cooldown.read_text()) > time.time()
            except (OSError, ValueError):
                waited_into_cooldown = False
            if waited_into_cooldown:
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()
                continue
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        return


def _cool_down(model: str) -> None:
    (_SLOT_DIR / f"{account_of(model)}.cooldown").write_text(str(time.time() + _COOLDOWN_SECONDS))


#: Models that refuse any ``temperature`` (Anthropic's newest answer HTTP 400 "`temperature` is
#: deprecated for this model"); learned on first refusal, so a judge is never silently retried.
_NO_TEMPERATURE: set[str] = set()


def _env_value(name: str) -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"')
    return os.environ.get(name, "")


def _openrouter_keys() -> list[str]:
    primary = _env_value("OPENROUTER_API_KEY")
    extra = _env_value("OPENROUTER_FALLBACK_API_KEYS").replace(",", " ").split()
    return [key for key in [primary, *extra] if key]


def _require_free(model_id: str) -> None:
    """Refuse an OpenRouter model whose catalog price is not zero, before any request is sent."""

    if model_id not in _FREE_ON_OPENROUTER:
        with urllib.request.urlopen(OPENROUTER_CATALOG, timeout=60) as response:
            catalog = json.loads(response.read())["data"]
        for entry in catalog:
            pricing = entry.get("pricing") or {}
            _FREE_ON_OPENROUTER[entry["id"]] = all(
                float(pricing.get(part) or 0) == 0 for part in ("prompt", "completion", "request")
            )
    if not _FREE_ON_OPENROUTER.get(model_id, False):
        raise PermanentJudgeError(f"{model_id} is not listed at zero price on OpenRouter")


def _openrouter_chat(
    model: str, prompt: str, *, max_tokens: int, temperature: float
) -> tuple[str, dict[str, Any]]:
    model_id = model.removeprefix(OPENROUTER_PREFIX)
    _require_free(model_id)
    base = (_env_value("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # A free route may retain prompts; the open subset is public (CC BY 4.0).
        "provider": {"data_collection": "allow"},
    }
    for key in _openrouter_keys():
        if key in _SPENT_OPENROUTER:
            continue
        request = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with account_slot(model), urllib.request.urlopen(request, timeout=600) as response:
                reply = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:600].decode("utf-8", "replace")
            if exc.code == 402 or "free-models-per-day" in detail or "free_tier_daily" in detail:
                _SPENT_OPENROUTER.add(key)  # this account's allowance, not the request
                continue
            if exc.code in (429, 503):
                _cool_down(model)
                raise
            if 400 <= exc.code < 500:
                raise PermanentJudgeError(f"HTTP {exc.code}: {detail[:300]}") from exc
            raise
        if "error" in reply and not reply.get("choices"):
            raise RuntimeError(f"OpenRouter error: {str(reply['error'])[:300]}")
        message = reply["choices"][0].get("message") or {}
        usage = reply.get("usage") or {}
        return str(message.get("content") or ""), {
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "thinking_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens", 0
            )
            or 0,
        }
    raise RuntimeError("every configured OpenRouter account has spent today's free allowance")


def chat(
    model: str, prompt: str, *, max_tokens: int, temperature: float
) -> tuple[str, dict[str, Any]]:
    if model.startswith(OPENROUTER_PREFIX):
        return _openrouter_chat(model, prompt, max_tokens=max_tokens, temperature=temperature)
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if model not in _NO_TEMPERATURE:
        body["temperature"] = temperature
    request = urllib.request.Request(
        f"{ENDPOINT}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {endpoint_key()}", "Content-Type": "application/json"},
    )
    try:
        with account_slot(model), urllib.request.urlopen(request, timeout=600) as response:
            reply = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300].decode("utf-8", "replace")
        if exc.code in (429, 503):
            _cool_down(model)
        if exc.code == 400 and "temperature" in detail and model not in _NO_TEMPERATURE:
            _NO_TEMPERATURE.add(model)
            return chat(model, prompt, max_tokens=max_tokens, temperature=temperature)
        if 400 <= exc.code < 500 and exc.code != 429:
            raise PermanentJudgeError(f"HTTP {exc.code}: {detail}") from exc
        raise
    message = reply["choices"][0].get("message") or {}
    usage = reply.get("usage") or {}
    return str(message.get("content") or ""), {
        "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
        "completion_tokens": usage.get("completion_tokens", 0) or 0,
        "thinking_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        or 0,
    }


def judge_call_raw(
    model_id: str,
    prompt: str,
    schema: dict[str, Any],
    *,
    max_output_tokens: int = 32768,
    temperature: float = 0.0,
    **_: Any,
) -> tuple[str, float, dict[str, Any]]:
    """The kit's ``gemini_sdk.sync_call_raw`` contract, answered by an allowed model.

    Gemini is given the schema as ``response_schema``; an OpenAI-compatible endpoint is given it in
    the prompt, and the reply is validated here, so a malformed reply is re-asked rather than
    handed to the kit's stages as a verdict.
    """

    import importlib

    jsonschema: Any = importlib.import_module("jsonschema")

    model = model_id.removeprefix("local/")
    instruction = (
        f"{prompt}\n\nReturn only one JSON object, with no prose and no code fence, that validates "
        f"against this JSON Schema:\n{json.dumps(schema)}"
    )
    started = time.time()
    last_error: Exception | None = None
    usage: dict[str, Any] = {}
    for attempt in range(8):
        try:
            text, usage = chat(
                model, instruction, max_tokens=max_output_tokens, temperature=temperature
            )
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            # The first complete object: a reply may carry a second object or a trailing remark.
            start = cleaned.find("{")
            parsed, _end = json.JSONDecoder().raw_decode(cleaned[start:] if start >= 0 else cleaned)
            jsonschema.validate(parsed, schema)
            return json.dumps(parsed), time.time() - started, usage
        except (urllib.error.URLError, TimeoutError, ValueError, jsonschema.ValidationError) as exc:
            last_error = exc
            print(
                f"[judge {model}] attempt {attempt + 1} failed: {str(exc)[:160]}", file=sys.stderr
            )
            time.sleep(min(120, 10 * 2**attempt))
    raise RuntimeError(f"judge {model} gave no schema-valid reply: {last_error}")


def install_judge() -> None:
    sys.path.insert(0, str(KIT))
    import importlib

    sdk: Any = importlib.import_module("judge.gemini_sdk")

    sdk.sync_call_raw = judge_call_raw
    sdk.is_gemini = lambda _model: True


def _kit(script: str, argv: list[str]) -> None:
    os.chdir(KIT)
    sys.argv = [script, *argv]
    runpy.run_path(str(KIT / script), run_name="__main__")


def cmd_run(args: argparse.Namespace) -> None:
    config = {
        "model": {
            "name": args.name,
            "platform": "local",
            "model_id": args.model,
            "api_base": ENDPOINT,
            # Reasoning models on the proxy refuse a temperature; the kit strips a key set to null.
            **({"completion_kwargs": {"temperature": None}} if args.no_temperature else {}),
        }
    }
    # litellm's OpenAI-compatible route reads the key from the environment, so it is never written
    # into the config file on disk.
    os.environ["OPENAI_API_KEY"] = endpoint_key()
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        json.dump(config, handle)  # JSON is YAML
    _kit(
        "run.py",
        [
            "--model-config",
            handle.name,
            "--benchmark-config",
            str(KIT / "config/benchmark.yaml"),
            "--k",
            str(args.k),
            "--threads",
            str(args.threads),
            *(["--prompt", args.prompt] if args.prompt else []),
        ],
    )


def cmd_score(args: argparse.Namespace) -> None:
    install_judge()
    config = {"model": {"name": args.name, "platform": "local", "model_id": args.name}}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        json.dump(config, handle)
    argv = [
        "--model-config",
        handle.name,
        "--benchmark-config",
        str(KIT / "config/benchmark.yaml"),
        "--k",
        str(args.k),
        "--threads",
        str(args.threads),
        "--match-judge",
        f"local/{args.judge}",
        "--review-judge",
        f"local/{args.review_judge or args.judge}",
    ]
    if args.cases:
        argv += ["--cases", args.cases]
    if args.prompt:
        argv += ["--prompt", args.prompt]
    _kit("score.py", argv)


def judged_metric(name: str, metric: str, prompt: str | None) -> dict[str, float]:
    """``variant id -> metric`` from an arm's judged file (the kit's per-record ``metrics``)."""

    path = (
        ROOT / "sources/mast/results/raw/donoharm" / (prompt or "default") / f"{name}_judged.jsonl"
    )
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            value = (row.get("metrics") or {}).get(metric)
            if isinstance(value, (int, float)):
                values[row["id"]] = float(value)
    return values


def cmd_compare(args: argparse.Namespace) -> None:
    """Paired difference of two arms over the variants both scored, bootstrapping whole cases.

    Variants of one case are not independent (they are perturbations of it), so the resampling
    unit is the case, as in the kit's own intervals.
    """

    import random

    left = judged_metric(args.a, args.metric, args.a_prompt)
    right = judged_metric(args.b, args.metric, args.b_prompt)
    common = sorted(set(left) & set(right))
    by_case: dict[str, list[float]] = {}
    for vid in common:
        by_case.setdefault(vid.split("-")[0], []).append(right[vid] - left[vid])
    cases = sorted(by_case)

    def mean_of(chosen: list[str]) -> float:
        diffs = [d for case in chosen for d in by_case[case]]
        return sum(diffs) / len(diffs)

    observed = mean_of(cases)
    rng = random.Random(7)
    boots = sorted(mean_of([rng.choice(cases) for _ in cases]) for _ in range(args.resamples))
    low, high = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    worse = sum(b <= 0 for b in boots) / len(boots)
    print(
        json.dumps(
            {
                "metric": args.metric,
                "a": args.a,
                "b": args.b,
                "variants": len(common),
                "cases": len(cases),
                "mean_a": round(sum(left[v] for v in common) / len(common), 4),
                "mean_b": round(sum(right[v] for v in common) / len(common), 4),
                "b_minus_a": round(observed, 4),
                "ci95": [round(low, 4), round(high, 4)],
                "bootstrap_share_b_not_better": round(worse, 4),
            },
            indent=1,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="generate plans with one allowed model")
    run.add_argument("--model", required=True)
    run.add_argument("--name", required=True, help="the kit's output namespace for this arm")
    run.add_argument("--k", type=int, default=11)
    run.add_argument("--threads", type=int, default=8)
    run.add_argument("--prompt", help="a prompt in the kit's prompts/ (default: unprompted)")
    run.add_argument("--no-temperature", action="store_true", help="for models that refuse one")
    run.set_defaults(func=cmd_run)
    score = sub.add_parser("score", help="judge and score with the kit, judge swapped")
    score.add_argument("--name", required=True)
    score.add_argument("--judge", required=True, help="an allowed model for the match stage")
    score.add_argument("--review-judge", help="an allowed model for the review stage")
    score.add_argument("--k", type=int, default=11)
    score.add_argument("--threads", type=int, default=8)
    score.add_argument("--cases", help="comma-separated base case ids")
    score.add_argument("--prompt", help="the prompt the arm was run with")
    score.set_defaults(func=cmd_score)
    compare = sub.add_parser("compare", help="paired difference of two scored arms")
    compare.add_argument("--a", required=True)
    compare.add_argument("--b", required=True)
    compare.add_argument("--a-prompt")
    compare.add_argument("--b-prompt")
    compare.add_argument("--metric", default="F1_weighted")
    compare.add_argument("--resamples", type=int, default=10000)
    compare.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
