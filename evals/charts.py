"""Figures for the evaluation page of the docs, drawn as SVG from `results.json` with the standard library only.

One message per figure, labels on the data instead of axes and legends, one accent for what matters.
"""

import json
from collections.abc import Callable
from html import escape
from pathlib import Path
from statistics import median
from typing import Any

RESULTS = Path(__file__).parent / "results.json"
WIDTH = 640
LABEL = 250  # width of the label column


def load() -> dict[str, Any]:
    return dict(json.loads(RESULTS.read_text()))


def graded(results: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in results["scenarios"] if s["checks"]]


def svg(height: float, body: str, label: str) -> str:
    return (
        f'<svg class="eval-figure" viewBox="0 0 {WIDTH} {height:.0f}" role="img" aria-label="{escape(label)}">'
        f"{body}</svg>"
    )


def text(x: float, y: float, content: str, cls: str = "", anchor: str = "start") -> str:
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{escape(content)}</text>'


def lead(results: dict[str, Any]) -> str:
    scenarios = graded(results)
    trials = [t for s in scenarios for t in s["trials"]]
    passed = sum(t["passed"] for t in trials)
    return (
        f"Every scenario ran {results['repeat']} times on the Foundation Model, graded on the path it took: tool calls"
        f" and arguments, questions to the user, rounds and answer. The bridge held in **{passed} of {len(trials)}**"
        " trials. What is worth a look is how the model behaves, and where the time goes."
    )


def scoreboard(results: dict[str, Any]) -> str:
    """A row of trials per scenario: filled when every check held."""
    rows, step = graded(results), 30
    body = []
    for i, s in enumerate(rows):
        y = 20 + i * step
        body.append(text(0, y + 5, s["title"], "label"))
        for j, t in enumerate(s["trials"]):
            body.append(
                f'<circle class="{"held" if t["passed"] else "broke"}" cx="{LABEL + 8 + j * 18}" cy="{y}" r="6"/>'
            )
        n, ok = len(s["trials"]), sum(t["passed"] for t in s["trials"])
        body.append(text(LABEL + 8 + n * 18, y + 5, f"{ok} of {n}", "value"))
    return svg(len(rows) * step + 10, "".join(body), "Trials per scenario in which every check held")


def findings(results: dict[str, Any]) -> str:
    """How often the model did something, as bars out of all trials, labelled above the bar."""
    rows = [(s["title"], name, s["trials"]) for s in results["scenarios"] for name in s["findings"]]
    step, track = 58, WIDTH - 90
    body = []
    for i, (title, name, trials) in enumerate(rows):
        y, seen = 4 + i * step, sum(t["findings"][name] for t in trials)
        body.append(
            f'<text x="0" y="{y + 14}"><tspan class="label">{escape(name[0].upper() + name[1:])}</tspan>'
            f'<tspan class="muted"> · {escape(title)}</tspan></text>'
        )
        body.append(f'<rect class="track" x="0" y="{y + 22}" width="{track}" height="20" rx="2"/>')
        body.append(
            f'<rect class="accent" x="0" y="{y + 22}" width="{track * seen / len(trials):.1f}" height="20" rx="2"/>'
        )
        body.append(text(track + 12, y + 37, f"{seen} of {len(trials)}", "value"))
    return svg(len(rows) * step, "".join(body), "How often the model showed each behaviour")


def seconds(span: dict[str, Any]) -> float:
    return float(span["end_ms"] - span["start_ms"]) / 1000


def time_split(results: dict[str, Any]) -> str:
    """The median trial of each scenario, one block per Apple generation: more rounds, more time."""
    rows = [(s["title"], median_trial(s)) for s in graded(results)]
    rows.sort(key=duration, reverse=True)
    longest, step, track = max(float(t["duration_s"]) for _, t in rows), 30, WIDTH - LABEL - 70
    body = []
    for i, (title, trial) in enumerate(rows):
        y, scale = 6 + i * step, track / (longest * 1000)
        body.append(text(0, y + 15, title, "label"))
        body.extend(
            f'<rect class="gen" x="{LABEL + sp["start_ms"] * scale:.1f}" y="{y + 2}" width="{max(2.0, seconds(sp) * 1000 * scale - 2):.1f}" height="18" rx="2"/>'
            for sp in trial["spans"]
            if sp["name"].startswith("chat")
        )
        body.append(
            text(LABEL + trial["duration_s"] * 1000 * scale + 8, y + 15, f"{trial['duration_s']:.1f} s", "value")
        )
    return svg(len(rows) * step + 4, "".join(body), "Median trial per scenario, one block per Apple generation")


def duration(row: tuple[str, dict[str, Any]]) -> float:
    return float(row[1]["duration_s"])


def closest_to(target: float) -> Callable[[dict[str, Any]], float]:
    def distance(trial: dict[str, Any]) -> float:
        return abs(float(trial["duration_s"]) - target)

    return distance


def median_trial(scenario: dict[str, Any]) -> dict[str, Any]:
    trials: list[dict[str, Any]] = [t for t in scenario["trials"] if "error" not in t]
    middle = min(trials, key=closest_to(median(t["duration_s"] for t in trials)))
    return middle


def protocols(results: dict[str, Any]) -> str:
    """The median trial of each protocol on one time axis, with numbered moments that the text explains."""
    by_id = {s["id"]: s for s in results["scenarios"]}
    tracks = [
        ("Classic", median_trial(by_id["elicitation-classic"])),
        ("Stateless", median_trial(by_id["elicitation-stateless"])),
    ]
    end = float(max(sp["end_ms"] for _, t in tracks for sp in t["spans"]))

    def x(ms: float) -> float:
        return LABEL + ms / end * (WIDTH - LABEL - 16)

    def callout(ms: float, y: float, number: int) -> str:
        return f'<line class="leader" x1="{x(ms):.1f}" x2="{x(ms):.1f}" y1="{y + 11}" y2="{y + 30}"/><circle class="callout" cx="{x(ms):.1f}" cy="{y}" r="10"/>{text(x(ms), y + 4, str(number), "callout-n", "middle")}'

    body, step = [], 84
    for i, (name, trial) in enumerate(tracks):
        y, spans = 36 + i * step, trial["spans"]
        body.append(text(0, y + 13, name, "label"))
        body.append(
            text(
                0,
                y + 31,
                f"{trial['duration_s']:.1f} s · {trial['agent_runs']} round"
                + ("" if trial["agent_runs"] == 1 else "s"),
                "muted",
            )
        )
        body.append(f'<line class="baseline" x1="{LABEL}" x2="{WIDTH - 16}" y1="{y + 14}" y2="{y + 14}"/>')
        body.extend(
            f'<rect class="gen" x="{x(sp["start_ms"]):.1f}" y="{y + 4}" width="{max(1.5, x(sp["end_ms"]) - x(sp["start_ms"])):.1f}" height="20" rx="2"/>'
            for sp in spans
            if sp["name"].startswith("chat")
        )
        body.extend(
            f'<rect class="tool" x="{x(sp["start_ms"]) - 1:.1f}" y="{y + 1}" width="3" height="26"/>'
            for sp in spans
            if sp["name"].startswith(("execute_tool", "replay_tool"))
        )
        asked = [sp for sp in spans if sp["name"] == "MCP send elicitation/create"]
        calls = [sp for sp in spans if sp["name"].startswith("tools/call")]
        replayed = [sp for sp in spans if sp["name"].startswith("replay_tool")]
        if asked:
            body.append(callout(asked[0]["start_ms"], y - 22, 1))
        if len(calls) > 1:
            body.append(
                f'<line class="boundary" x1="{x(calls[0]["end_ms"]):.1f}" x2="{x(calls[0]["end_ms"]):.1f}" y1="{y - 2}" y2="{y + 30}"/>'
            )
            body.append(callout(calls[0]["end_ms"], y - 22, 2))
        if replayed:
            body.append(callout(replayed[0]["start_ms"], y - 22, 3))
        if len(calls) > 1:
            body.append(
                text((x(calls[0]["start_ms"]) + x(calls[0]["end_ms"])) / 2, y + 42, "round 1", "muted", "middle")
            )
            body.append(
                text(
                    (x(calls[1]["start_ms"]) + x(calls[1]["end_ms"])) / 2,
                    y + 42,
                    "round 2, regenerated",
                    "muted",
                    "middle",
                )
            )
    ticks = range(0, int(end) + 1, 1000)
    body.extend(text(x(ms), 36 + len(tracks) * step - 10, f"{ms // 1000} s", "tick", "middle") for ms in ticks)
    return svg(
        36 + len(tracks) * step,
        "".join(body),
        "One question through the classic and the stateless protocol, on one time axis",
    )
