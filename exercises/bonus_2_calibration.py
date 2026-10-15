"""Bonus 2 — Confidence calibration from the audit trail.

The LLM's `confidence` claims "probability this review is complete and the PR
is fine as-is". The human decisions in `audit_events` tell us how often that
was actually true. Comparing the two per confidence bucket shows whether the
model is over- or under-confident.

Label per session (taken from audit_events):
    accepted     = reviewer chose "approve" on the approval card
    not accepted = reviewer chose "reject" / "edit", or the session escalated
    (auto-approved sessions have no human verdict and are reported separately)

    uv run python exercises/bonus_2_calibration.py                   # report on existing sessions
    uv run python exercises/bonus_2_calibration.py --simulate 10     # first run 10 dry-run sessions

`--simulate` drives the exercise-4 graph with a scripted reviewer whose
verdict per PR is given by --policy (default: PR #1 approve, PR #2 reject).
Simulated sessions never post to GitHub.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from rich.console import Console
from rich.table import Table

from common.db import db_conn, db_path
from common.github import parse_pr_url


console = Console()
DEMO = "https://github.com/VinUni-AI20k/PR-Demo/pull/{}"

# One row per session: first LLM confidence + the human verdict, if any.
SESSIONS_SQL = """
WITH first_analyze AS (
    SELECT thread_id, pr_url, confidence
      FROM audit_events
     WHERE id IN (SELECT MIN(id) FROM audit_events WHERE action = 'analyze' GROUP BY thread_id)
),
verdict AS (
    SELECT thread_id,
           MAX(CASE WHEN action = 'human_approval' AND decision = 'approve' THEN 1 ELSE 0 END) AS approved,
           MAX(CASE WHEN action = 'human_approval' AND decision IN ('reject', 'edit') THEN 1 ELSE 0 END) AS refused,
           MAX(CASE WHEN action = 'escalate' AND decision = 'answered' THEN 1 ELSE 0 END) AS escalated,
           MAX(CASE WHEN action = 'auto_approve' THEN 1 ELSE 0 END) AS auto
      FROM audit_events GROUP BY thread_id
)
SELECT f.thread_id, f.pr_url, f.confidence, v.approved, v.refused, v.escalated, v.auto
  FROM first_analyze f JOIN verdict v USING (thread_id)
"""

BUCKETS = [(0.0, 0.58, "< 58%  (escalate)"), (0.58, 0.73, "58-72% (human)"), (0.73, 1.01, ">= 73% (auto)")]


async def report() -> None:
    async with db_conn() as conn:
        async with conn.execute(SESSIONS_SQL) as cur:
            rows = [dict(r) for r in await cur.fetchall()]

    labelled = [r | {"accepted": r["approved"]} for r in rows
                if r["approved"] or r["refused"] or r["escalated"]]
    auto = [r for r in rows if r["auto"]]
    console.print(f"{len(rows)} sessions with an analysis · {len(labelled)} with a human verdict · "
                  f"{len(auto)} auto-approved (no human verdict)\n")
    if not labelled:
        console.print("[yellow]No human-labelled sessions yet — run with --simulate N.[/yellow]")
        return

    t = Table(title="Calibration: stated confidence vs. observed acceptance")
    for col in ("bucket", "n", "avg confidence", "accepted as-is", "gap (conf - accepted)"):
        t.add_column(col, justify="right")
    for lo, hi, name in BUCKETS:
        b = [r for r in labelled if lo <= r["confidence"] < hi]
        if not b:
            t.add_row(name, "0", "-", "-", "-")
            continue
        conf = sum(r["confidence"] for r in b) / len(b)
        acc = sum(r["accepted"] for r in b) / len(b)
        gap = conf - acc
        color = "red" if gap > 0.1 else "cyan" if gap < -0.1 else "green"
        t.add_row(name, str(len(b)), f"{conf:.2f}", f"{acc:.2f}", f"[{color}]{gap:+.2f}[/]")
    console.print(t)

    per_pr = Table(title="Per PR")
    for col in ("pr", "n", "avg confidence", "min-max", "accepted as-is"):
        per_pr.add_column(col, justify="right")
    for pr in sorted({r["pr_url"] for r in labelled}):
        b = [r for r in labelled if r["pr_url"] == pr]
        cs = [r["confidence"] for r in b]
        per_pr.add_row(f"#{parse_pr_url(pr)[2]}", str(len(b)), f"{sum(cs) / len(cs):.2f}",
                       f"{min(cs):.2f}-{max(cs):.2f}", f"{sum(r['accepted'] for r in b) / len(b):.2f}")
    console.print(per_pr)

    brier = sum((r["confidence"] - r["accepted"]) ** 2 for r in labelled) / len(labelled)
    mean_gap = sum(r["confidence"] - r["accepted"] for r in labelled) / len(labelled)
    verdict = ("over-confident" if mean_gap > 0.05 else
               "under-confident" if mean_gap < -0.05 else "roughly calibrated")
    console.print(f"\nBrier score = [bold]{brier:.3f}[/bold] (0 = perfect, 0.25 = coin flip)   "
                  f"mean(confidence - accepted) = [bold]{mean_gap:+.2f}[/bold] → model is [bold]{verdict}[/bold]")


async def simulate(n: int, policy: dict[int, str]) -> None:
    from exercises.exercise_4_audit import build_graph

    os.environ["HITL_DRY_RUN"] = "1"
    prs = sorted(policy)
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        for i in range(n):
            pr = prs[i % len(prs)]
            tid = str(uuid.uuid4())
            cfg = {"configurable": {"thread_id": tid}}
            console.rule(f"[dim]simulated session {i + 1}/{n} — PR #{pr} (reviewer policy: {policy[pr]})")
            result = await app.ainvoke({"pr_url": DEMO.format(pr), "thread_id": tid}, cfg)
            while "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                if payload["kind"] == "approval_request":
                    answer = {"choice": policy[pr], "feedback": "simulated reviewer"}
                else:
                    answer = {q: "Simulated reviewer: this needs to be fixed before merge."
                              for q in payload["questions"]}
                result = await app.ainvoke(Command(resume=answer), cfg)


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--simulate", type=int, default=0, help="Run N dry-run sessions first")
    p.add_argument("--policy", nargs="+", default=["1=approve", "2=reject"],
                   help="Scripted reviewer verdict per demo PR, e.g. 1=approve 2=reject")
    args = p.parse_args()
    if args.simulate:
        policy = {int(k): v for k, v in (item.split("=") for item in args.policy)}
        asyncio.run(simulate(args.simulate, policy))
    asyncio.run(report())


if __name__ == "__main__":
    main()
