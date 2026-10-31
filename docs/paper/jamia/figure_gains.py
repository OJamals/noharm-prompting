"""Figure 1 of version 2: what the instruction changed, and which part changed it.

Every estimate is read from the macro files the manuscript inputs, so the figure cannot drift
from the text. Colors match the paper's other figures.

    .venv/bin/python docs/paper/jamia/figure_gains.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "fig-gains.pdf"

PROMPT, INK, MUTED, RULE = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"

MACRO = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$")
INTERVAL = re.compile(r"([+\-]?\d*\.\d+)\\,\[([+\-]?\d*\.\d+),\\,([+\-]?\d*\.\d+)\]")


def macros() -> dict[str, str]:
    """Every \\newcommand value in the files the manuscript inputs."""
    found: dict[str, str] = {}
    for name in ("numbers.tex", "coverage-numbers.tex", "coverage-analyses.tex"):
        for line in (PAPER / name).read_text(encoding="utf-8").splitlines():
            m = MACRO.match(line.strip())
            if m:
                found[m.group(1)] = m.group(2)
    return found


def interval(value: str) -> tuple[float, float, float]:
    """(estimate, low, high) from a paired-difference macro; a pending one is an error."""
    m = INTERVAL.search(value)
    if not m:
        raise ValueError(f"not an estimate with an interval: {value!r}")
    est, lo, hi = (float(g) for g in m.groups())
    return est, lo, hi


def forest(ax, rows, xlim, title):
    """One row per estimate; filled where the interval excludes zero, hollow where it does not."""
    for y, (_label, (est, lo, hi)) in enumerate(rows):
        clear = lo > 0 or hi < 0
        color = PROMPT if clear else MUTED
        ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(
            est,
            y,
            s=64,
            zorder=3,
            facecolor=color if clear else "white",
            edgecolor=color,
            linewidth=2,
        )
        ax.text(
            1.03,
            y,
            f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}]",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows], fontsize=9, color=INK)
    ax.set_ylim(len(rows) - 0.5, -0.7)
    ax.set_xlim(*xlim)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=RULE, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_title(title, loc="left", fontsize=10, weight="bold", color=INK)


def main() -> None:
    m = macros()
    # Ordered by how much each model wrote on the bare case, tersest first.
    models = [
        ("Grok 4.7", "Grok", "covDPromptGrokThorough"),
        ("GPT-5.6 Terra", "Terra", "covDPromptTerra"),
        ("GPT-6 Astra", "Astra", "covDPromptAstraExh"),
        ("Claude Sonnet 5.5", "Sonnet", "covDPromptSonnet"),
        ("Muse Spark 1.3", "Muse", "covDPromptMuseThorough"),
    ]
    gains = [
        (f"{name} ({m[f'covA{alias}Bare']} actions bare)", interval(m[key]))
        for name, alias, key in models
    ]
    parts = [
        ("Plan format only: Sonnet 5.5", interval(m["covDSonnetFormatVsSpec"])),
        ("Terra", interval(m["covDTerraFormatVsSpec"])),
        ("Consult framing only: Sonnet 5.5", interval(m["covDSonnetFramingVsSpec"])),
        ("Terra", interval(m["covDTerraFramingVsSpec"])),
        ("25 recommendations: Sonnet 5.5", interval(m["covDSonnetLengthVsSpec"])),
        ("Terra", interval(m["covDTerraLengthVsSpec"])),
    ]

    plt.rcParams.update({"font.family": "Helvetica", "font.size": 9})
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.4),
        gridspec_kw={"height_ratios": [len(gains), len(parts)], "hspace": 0.55},
    )
    forest(top, gains, (-0.06, 0.42), "A. Instructed minus bare case, by model")
    forest(bottom, parts, (-0.16, 0.04), "B. One part of the instruction alone minus the full one")
    bottom.set_xticks([-0.15, -0.10, -0.05, 0.0])
    for ax in (top, bottom):
        ax.set_xlabel("Difference in severity-weighted F1", color=INK, fontsize=9)
    fig.text(
        0.01,
        0.01,
        "Paired by case, 95% intervals from 10,000 case resamples. Filled: interval excludes zero.",
        fontsize=7.5,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.36, right=0.74, top=0.94, bottom=0.12)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=200, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
