"""Build the project page from the manuscript's own generated numbers.

    python site/build.py            # writes site/index.html

Every figure on the page is read from docs/paper/coverage-numbers.tex and
docs/paper/coverage-analyses.tex (written by scripts/noharm_paper_snapshot.py and
scripts/noharm_paper_analyses.py), and the instructions from scripts/noharm_system.py, so the
page cannot drift from the manuscript. A value the manuscript marks pending stops the build.
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.noharm_system import ABLATIONS, PLAN_FORMAT, SPECIALIST  # noqa: E402

SITE = ROOT / "site"
PAPER = ROOT / "docs/paper"

#: Filled in once the preprint is posted; until then the page links the PDF in the archive.
PREPRINT_URL = ""
REPO_URL = "https://github.com/OJamals/noharm-prompting"
PDF_URL = REPO_URL + "/raw/main/docs/paper/coverage.pdf"
SITE_URL = "https://noharm.aljamal.work"

#: (display name, bare arm alias, instructed arm alias, instruction label, gain macro)
MODELS: tuple[tuple[str, str, str | None, str, str | None], ...] = (
    ("Claude Sonnet 5.5", "SonnetBare", "SonnetSpec", "specialist", "DPromptSonnet"),
    ("GPT-5.6 terra", "TerraBare", "TerraSpec", "specialist", "DPromptTerra"),
    ("GPT-6 Astra", "AstraBare", "AstraExh", "exhaustive", "DPromptAstraExh"),
    ("Grok 4.7", "GrokBare", "GrokThorough", "kit, 500 words", "DPromptGrokThorough"),
    ("Muse Spark 1.3", "MuseBare", "MuseThorough", "kit, 500 words", "DPromptMuseThorough"),
    ("Claude Opus 5.5", "OpusBare", None, "", None),
    ("Claude Sonnet 5", "SonnetOldBare", None, "", None),
    ("Claude Haiku 4.5", "HaikuBare", None, "", None),
    ("GPT-5.6 Luna", "LunaBare", None, "", None),
    ("GPT-6.1 Sol", "SolBare", None, "", None),
    ("Nemotron 3 Ultra", "NemotronBare", None, "", None),
    ("Big Pickle", "PickleBare", None, "", None),
)
COMPONENTS = (
    ("Bare", "Bare"),
    ("Framing only", "Framing"),
    ("Length only", "Length"),
    ("Format only", "Format"),
    ("Full instruction", "Spec"),
)


def macros() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("coverage-numbers.tex", "coverage-analyses.tex"):
        for line in (PAPER / name).read_text(encoding="utf-8").splitlines():
            match = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", line)
            if match:
                out[match.group(1)] = match.group(2)
    return out


M = macros()


def raw(name: str) -> str:
    value = M["cov" + name]
    if "covPending" in value:
        raise SystemExit(f"cov{name} is pending in the manuscript; the page cannot quote it")
    return value


def clean(value: str) -> str:
    value = value.replace(r"\ensuremath{", "").replace(r"\,", " ").replace("{,}", ",")
    return value.replace(r"\ensuremath{<}", "<").rstrip("}").strip()


def num(name: str) -> float:
    return float(clean(raw(name)).replace(",", ""))


def interval(name: str) -> list[float]:
    low, high = re.findall(r"[-+]?\d*\.?\d+", clean(raw(name)))
    return [float(low), float(high)]


def diff(name: str) -> dict[str, float]:
    mean, low, high = re.findall(r"[-+]?\d*\.?\d+", clean(raw(name)))
    return {"mean": float(mean), "lo": float(low), "hi": float(high)}


def arm(alias: str, with_ci: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {
        "f1": num("F" + alias),
        "f1all": num("FAll" + alias),
        "actions": num("A" + alias),
        "precision": num("P" + alias),
        "recall": num("R" + alias),
        "severe": num("S" + alias),
        "chars": num("Chars" + alias),
        "offrubric": num("Off" + alias),
    }
    if with_ci:
        out["ci"] = interval("CI" + alias)
    return out


def data() -> dict[str, Any]:
    models = []
    for name, bare, prompted, label, gain in MODELS:
        row: dict[str, Any] = {"name": name, "bare": arm(bare)}
        if prompted:
            row["instructed"] = {**arm(prompted), "instruction": label}
            row["gain"] = diff(gain)  # type: ignore[arg-type]
            row["gainAll"] = diff("DAll" + gain[1:])  # type: ignore[index]
        models.append(row)
    components = {}
    for model in ("Sonnet", "Terra"):
        rows = []
        for label, part in COMPONENTS:
            alias = model + part
            entry: dict[str, Any] = {"label": label, **arm(alias)}
            if part == "Spec":
                entry["vsBare"] = diff("DPrompt" + model)
            elif part != "Bare":
                entry["vsBare"] = diff(f"D{alias}VsBare")
                entry["vsSpec"] = diff(f"D{alias}VsSpec")
            rows.append(entry)
        components[model] = rows
    return {
        "models": models,
        "components": components,
        "headline": {
            "meanGain": num("PaperGridMeanGain"),
            "meanGainCI": interval("PaperGridMeanGainCI"),
            "sdBare": num("PaperGridSdBare"),
            "sdSpec": num("PaperGridSdSpec"),
            "rangeBare": num("PaperGridRangeBare"),
            "rangeSpec": num("PaperGridRangeSpec"),
            "kendall": num("PaperGridKendall"),
            "rhoGainActions": num("PaperGridRhoGainActions"),
            "lengthRho": num("LengthRho"),
            "lengthRhoCI": interval("LengthRhoCI"),
            "lengthArms": num("LengthArms"),
            "adjShare": num("AdjShare"),
            "adjShareCI": interval("AdjShareCI"),
            "spreadModels": num("PaperGridModels"),
        },
        "systems": {
            "terraVsGap": diff("DTerraSpecVsGap"),
            "unionPrompt": num("FUnionPrompt"),
            "unionPromptJT": num("FUnionPromptJT"),
        },
        "evaluators": {
            "terraGainNemotron": diff("DPromptTerraJF"),
            "terraInteractionNemotron": diff("InteractionTerraJF"),
            "terraSpecUnderTerra": num("FTerraSpecJT"),
        },
        "safety": {
            "harm": {
                m: {"bare": num(f"Harm{m}Bare"), "prompt": num(f"Harm{m}Prompt"),
                    "diff": diff(f"Harm{m}Diff")}
                for m in ("Sonnet", "Terra", "Astra", "Grok")
            },
            "severeSonnet": diff("SevereHarmSonnetDiff"),
            "audit": {
                m: {
                    "actionsBare": num(f"AuditActionsBare{m}"),
                    "actionsPrompt": num(f"AuditActionsPrompt{m}"),
                    "harmDiff": diff(f"AuditHarmDiff{m}"),
                    "okBare": num(f"RatedOk{m}NemoBare"),
                    "okPrompt": num(f"RatedOk{m}NemoPrompt"),
                }
                for m in ("Sonnet", "Terra", "Astra", "Grok")
            },
            "lingHarmDiffSonnet": diff("AuditHarmDiffSonnetLing"),
            "raw": {m: diff(f"DRaw{m}") for m in ("Sonnet", "Terra", "Astra", "Grok")},
            "physician": {
                "actions": num("PhysActions"),
                "flagged": num("PhysFlagged"),
                "controls": num("PhysControls"),
                "inappropriate": num("PhysInappropriate"),
                "options": num("PhysOptions"),
                "statements": num("PhysStatements"),
                "harmful": num("PhysHarmful"),
                "instructions": num("PhysInstructions"),
                "concerns": num("PhysConcerns"),
            },
        },
        "instructions": {
            "full": SPECIALIST + "\n\nCase:\n{case}\n\n" + PLAN_FORMAT,
            "format": ABLATIONS["format"],
        },
        "links": {
            "preprint": PREPRINT_URL,
            "pdf": PDF_URL,
            "repo": REPO_URL,
            "site": SITE_URL,
        },
    }


def main() -> None:
    payload = json.dumps(data(), ensure_ascii=False, indent=1).replace("</", "<\\/")
    template = (SITE / "template.html").read_text(encoding="utf-8")
    page = template.replace("/*DATA*/null", payload)
    for key, value in (("PDF_URL", PDF_URL), ("REPO_URL", REPO_URL), ("SITE_URL", SITE_URL)):
        page = page.replace("{{" + key + "}}", html.escape(value))
    (SITE / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {SITE / 'index.html'} ({len(page):,} bytes)")


if __name__ == "__main__":
    main()
