# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Automated Logbook Judge for SPD-Learn Reproducibility Auditing.

Modeled after the ICML 2026 Open Reproductions `logbook-judge` pipeline and the
NEMAR / MOABB / spd-bench harness protocol:
- Treats agent self-assessment narrative text as untrusted.
- Evaluates each claim in `repro/challenge.json` strictly from the machine-readable
  Trackio execution logbook (`repro/logbooks/<paper_id>_logbook.json`):
  1. Verifies `logbook["exit_code"] == 0`.
  2. Verifies all recorded numeric metrics in `claim_traces[claim_id]` are finite
     (`not math.isnan(v) and not math.isinf(v)`).
  3. Inspects the deterministic `passed` and `scope` fields from the execution trace:
     - `Verified` (2 pts): `exit_code == 0`, `passed is True`, all metrics finite,
       `scope == "full"` (full mathematical/signal verification or MOABB/spd-bench
       harness replication within CI).
     - `Baseline-Adjusted` (2 pts): `exit_code == 0`, `passed is True`, all metrics
       finite, `scope == "baseline_adjusted"` (executes on real MOABB/spd-bench
       data and confirms directional gain, while attributing headline gap to a
       verified baseline, split, or momentum discrepancy).
     - `Falsified` (2 pts): `exit_code == 0`, `passed is False`, all metrics finite.
     - `Toy` (1 pt): `exit_code == 0`, `passed is True`, `scope == "toy"`.
     - `Inconclusive` (0 pts): missing logbook, non-zero exit code, or NaN/Inf metric.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

REPRO_DIR = Path(__file__).resolve().parent
LOGBOOKS_DIR = REPRO_DIR / "logbooks"


def _all_metrics_finite(obj: Any) -> bool:
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, (int, bool, str)) or obj is None:
        return True
    if isinstance(obj, dict):
        return all(_all_metrics_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_all_metrics_finite(v) for v in obj)
    return True


def judge_claim(
    paper_id: str,
    anchor_claim: dict[str, Any],
    logbook: dict[str, Any] | None,
) -> dict[str, Any]:
    claim_id = anchor_claim["claim_id"]
    category = anchor_claim["category"]

    if logbook is None:
        return {
            "claim_id": claim_id,
            "paper_id": paper_id,
            "category": category,
            "verdict": "Inconclusive",
            "points": 0,
            "reason": "Missing Trackio logbook file.",
            "metrics": {},
        }

    if logbook.get("exit_code", -1) != 0:
        return {
            "claim_id": claim_id,
            "paper_id": paper_id,
            "category": category,
            "verdict": "Inconclusive",
            "points": 0,
            "reason": f"Non-zero execution exit code: {logbook.get('exit_code')}",
            "metrics": {},
        }

    claim_traces = logbook.get("claim_traces", {})
    if claim_id not in claim_traces:
        return {
            "claim_id": claim_id,
            "paper_id": paper_id,
            "category": category,
            "verdict": "Inconclusive",
            "points": 0,
            "reason": "Claim missing from logbook execution records.",
            "metrics": {},
        }

    rec = dict(claim_traces[claim_id])
    passed = rec.pop("passed", False)
    scope = rec.pop("scope", "full")
    metrics = rec

    if not _all_metrics_finite(metrics) or not metrics:
        return {
            "claim_id": claim_id,
            "paper_id": paper_id,
            "category": category,
            "verdict": "Inconclusive",
            "points": 0,
            "reason": "Non-finite (NaN/Inf) or empty metrics in execution trace.",
            "metrics": metrics,
        }

    if passed is True:
        if scope == "full":
            return {
                "claim_id": claim_id,
                "paper_id": paper_id,
                "category": category,
                "verdict": "Verified",
                "points": 2,
                "reason": "Verified on full manifold/signal invariant or MOABB/spd-bench harness within CI.",
                "metrics": metrics,
            }
        if scope == "baseline_adjusted":
            return {
                "claim_id": claim_id,
                "paper_id": paper_id,
                "category": category,
                "verdict": "Baseline-Adjusted",
                "points": 2,
                "reason": "Verified directional effect on MOABB/spd-bench harness with mechanistic attribution of baseline/protocol delta.",
                "metrics": metrics,
            }
        return {
            "claim_id": claim_id,
            "paper_id": paper_id,
            "category": category,
            "verdict": "Toy",
            "points": 1,
            "reason": "Reproduced on reduced-scale synthetic benchmark.",
            "metrics": metrics,
        }
    return {
        "claim_id": claim_id,
        "paper_id": paper_id,
        "category": category,
        "verdict": "Falsified",
        "points": 2,
        "reason": "Deterministic execution completed cleanly with metrics refuting the anchor threshold.",
        "metrics": metrics,
    }


def run_judge() -> dict[str, Any]:
    papers_meta = {
        p["paper_id"]: p for p in json.loads((REPRO_DIR / "papers.json").read_text())
    }
    challenge = json.loads((REPRO_DIR / "challenge.json").read_text())

    verdicts = []
    counts = {
        "Verified": 0,
        "Baseline-Adjusted": 0,
        "Falsified": 0,
        "Toy": 0,
        "Inconclusive": 0,
    }
    total_points = 0
    paper_scores: list[dict[str, Any]] = []

    for p_entry in challenge["papers"]:
        pid = p_entry["paper_id"]
        pmeta = papers_meta.get(pid, {})
        lb_path = LOGBOOKS_DIR / f"{pid}_logbook.json"
        lb = json.loads(lb_path.read_text()) if lb_path.exists() else None

        ps = {
            "paper_id": pid,
            "short_name": p_entry.get("name", pid),
            "venue": pmeta.get("venue", "N/A"),
            "year": pmeta.get("year", 2026),
            "points": 0,
            "max_points": 6,
            "verdicts": [],
        }

        for anchor in p_entry["claims"]:
            v = judge_claim(pid, anchor, lb)
            verdicts.append(v)
            counts[v["verdict"]] += 1
            total_points += v["points"]

            anchor["status"] = v["verdict"].upper().replace("-", "_")
            anchor["points"] = v["points"]

            ps["points"] += v["points"]
            ps["verdicts"].append(v)

        paper_scores.append(ps)

    challenge["status"] = "JUDGED"
    (REPRO_DIR / "challenge.json").write_text(json.dumps(challenge, indent=2))

    summary = {
        "judge_version": "spd-learn-logbook-judge-2.0",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_papers": len(paper_scores),
        "total_claims": len(verdicts),
        "verdict_counts": counts,
        "total_points": total_points,
        "max_possible_points": len(paper_scores) * 6,
        "paper_scores": paper_scores,
        "verdicts": verdicts,
    }
    (REPRO_DIR / "verdicts.json").write_text(json.dumps(summary, indent=2))

    md_lines = [
        "# SPD-Learn ICML 2026 / NEMAR / MOABB / spd-bench Reproductions Leaderboard",
        "",
        f"- **Judge**: `{summary['judge_version']}` (Untrusted-Narrative Execution Trace Auditor)",
        f"- **Audited Timestamp**: `{summary['timestamp_utc']}`",
        f"- **Papers Audited**: `{summary['total_papers']}` (8 Foundational + 18 Recent)",
        f"- **Claims Audited**: `{summary['total_claims']}` (`26 Structural`, `26 Signal`, `26 Analytical`)",
        f"- **Verdict Breakdown**: "
        f"`Verified: {counts['Verified']}` (2 pts) | "
        f"`Baseline-Adjusted: {counts['Baseline-Adjusted']}` (2 pts) | "
        f"`Falsified: {counts['Falsified']}` (2 pts) | "
        f"`Inconclusive: {counts['Inconclusive']}` (0 pts)",
        f"- **Total Reproducibility Score**: **`{total_points} / {summary['max_possible_points']}`**",
        "",
        "## Per-Paper Reproducibility Scorecard",
        "",
        "| Paper ID | Model / Method | Venue | Structural (C1) | Signal (C2) | Analytical (C3) | Points |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for ps in paper_scores:
        v_map = {v["category"]: v for v in ps["verdicts"]}
        c1 = v_map.get("Structural", {}).get("verdict", "N/A")
        c2 = v_map.get("Signal", {}).get("verdict", "N/A")
        c3 = v_map.get("Analytical", {}).get("verdict", "N/A")
        md_lines.append(
            f"| `{ps['paper_id']}` | **{ps['short_name']}** | {ps['venue']} "
            f"| `{c1}` | `{c2}` | `{c3}` | **{ps['points']} / 6** |"
        )

    md_lines.extend(
        [
            "",
            "## Detailed Claim Verdicts & Audited Metrics",
            "",
            "| Claim ID | Category | Verdict | Pts | Key Audited Metrics |",
            "| :--- | :--- | :---: | :---: | :--- |",
        ]
    )
    for v in verdicts:
        compact_metrics = ", ".join(
            f"`{k}={val:.4g}`" if isinstance(val, float) else f"`{k}={val}`"
            for k, val in v["metrics"].items()
        )
        md_lines.append(
            f"| `{v['claim_id']}` | {v['category']} | **{v['verdict']}** | {v['points']} | {compact_metrics} |"
        )

    (REPRO_DIR / "LEADERBOARD.md").write_text("\n".join(md_lines) + "\n")
    return summary


if __name__ == "__main__":
    res = run_judge()
    print(
        f"Judged {res['total_claims']} claims across {res['total_papers']} papers: "
        f"{res['verdict_counts']} | Total points: {res['total_points']} / {res['max_possible_points']}"
    )
