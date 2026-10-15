"""Bonus 3 — Multi-reviewer fan-out with the `Send` API.

On the escalate branch, the same questions go to several reviewers in
parallel: `route` fans out one `ask_reviewer` task per reviewer via `Send`.
Each task calls `interrupt()`, so the graph pauses with N pending interrupts
in the same super-step and waits for *all* of them. The caller resumes them
together with `Command(resume={interrupt_id: answers, ...})`. A `merge` node
then combines the answers and the regular exercise-4 `synthesize → commit`
path finishes the review.

    uv run python exercises/bonus_3_fanout.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
    uv run python exercises/bonus_3_fanout.py --pr ... --reviewers alice bob --auto-answer   # scripted

Runs in dry-run mode (no GitHub comment) unless --post is given.
"""

from __future__ import annotations

import argparse
import asyncio
import operator
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import db_path
from common.schemas import AuditEntry, ReviewState, risk_level_for
from exercises.exercise_4_audit import (
    AGENT_ID, _ms, audit, handle_interrupt,
    node_analyze, node_auto_approve, node_commit, node_fetch_pr,
    node_human_approval, node_revise, node_route, node_synthesize,
)


console = Console()


class FanoutState(ReviewState, total=False):
    reviewers: list[str]
    # Each ask_reviewer task appends one entry; operator.add merges parallel writes.
    reviewer_answers: Annotated[list[dict], operator.add]


def fan_out(state: FanoutState):
    """Conditional edge after route: one Send per reviewer on the escalate branch."""
    if state["decision"] != "escalate":
        return state["decision"]
    a = state["analysis"]
    questions = a.escalation_questions or ["What is the intent of this PR?"]
    return [
        Send("ask_reviewer", {
            "thread_id": state["thread_id"], "pr_url": state["pr_url"],
            "analysis": a, "reviewer": r, "questions": questions,
        })
        for r in state["reviewers"]
    ]


async def node_ask_reviewer(task: dict) -> dict:
    """Runs once per reviewer, in parallel. `task` is the Send payload, not the full state."""
    t0 = time.monotonic()
    a, reviewer = task["analysis"], task["reviewer"]
    answers = interrupt({
        "kind": "escalation",
        "reviewer": reviewer,
        "pr_url": task["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": task["questions"],
    })
    await audit(task, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=reviewer,
        decision="answered",
        reason=" | ".join(f"{q} -> {ans}" for q, ans in answers.items()),
        execution_time_ms=_ms(t0),
    ))
    return {"reviewer_answers": [{"reviewer": reviewer, "answers": answers}]}


async def node_merge(state: FanoutState) -> dict:
    """Join point: runs once after every reviewer has answered."""
    console.print(f"[cyan]→ merge[/cyan]  {len(state['reviewer_answers'])} reviewer(s) answered")
    merged: dict[str, str] = {}
    for entry in state["reviewer_answers"]:
        for q, ans in entry["answers"].items():
            merged.setdefault(q, "")
            merged[q] += f"{'; ' if merged[q] else ''}{entry['reviewer']}: {ans}"
    return {"escalation_answers": merged}


def build_graph(checkpointer):
    g = StateGraph(FanoutState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("revise", node_revise), ("commit", node_commit),
        ("ask_reviewer", node_ask_reviewer), ("merge", node_merge), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", fan_out, ["auto_approve", "human_approval", "ask_reviewer"],
    )
    g.add_edge("auto_approve", END)
    g.add_conditional_edges(
        "human_approval", lambda s: "revise" if s.get("human_choice") == "edit" else "commit",
        {"revise": "revise", "commit": "commit"},
    )
    g.add_edge("revise", "commit")
    g.add_edge("ask_reviewer", "merge")   # merge waits for all parallel ask_reviewer tasks
    g.add_edge("merge", "synthesize")
    g.add_edge("synthesize", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=checkpointer)


def scripted_answers(payload: dict) -> dict:
    return {q: "Not intended for production as written; please fix before merge."
            for q in payload["questions"]}


async def run(pr_url: str, reviewers: list[str], auto_answer: bool) -> None:
    thread_id = str(uuid.uuid4())
    console.rule("[bold]Bonus 3 — multi-reviewer fan-out[/bold]")
    console.print(f"[dim]PR: {pr_url}  reviewers: {reviewers}  thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        result = await app.ainvoke(
            {"pr_url": pr_url, "thread_id": thread_id, "reviewers": reviewers}, cfg,
        )
        while "__interrupt__" in result:
            pending = result["__interrupt__"]
            console.print(f"[yellow]Paused with {len(pending)} pending interrupt(s)[/yellow]")
            resume = {}
            for intr in pending:
                payload = intr.value
                if payload.get("reviewer"):
                    console.print(Panel.fit(payload["summary"], border_style="yellow",
                                            title=f"Reviewer {payload['reviewer']} — conf={payload['confidence']:.0%}"))
                    resume[intr.id] = scripted_answers(payload) if auto_answer else {
                        q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]
                    }
                else:   # approval_request when the PR lands in the 58-72% bucket
                    resume[intr.id] = handle_interrupt(payload)
            result = await app.ainvoke(Command(resume=resume), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        for q, ans in (result.get("escalation_answers") or {}).items():
            console.print(f"[bold]{q}[/bold]\n  {ans}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--reviewers", nargs="+", default=["reviewer-a", "reviewer-b"])
    p.add_argument("--auto-answer", action="store_true", help="Scripted answers (no prompts)")
    p.add_argument("--post", action="store_true", help="Actually post to GitHub (default: dry run)")
    args = p.parse_args()
    if not args.post:
        os.environ["HITL_DRY_RUN"] = "1"
    asyncio.run(run(args.pr, args.reviewers, args.auto_answer))


if __name__ == "__main__":
    main()
