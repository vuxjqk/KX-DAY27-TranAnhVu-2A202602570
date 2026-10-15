"""Bonus 1 — Time-travel.

List every checkpoint of a finished (or paused) exercise-4 thread with
`app.aget_state_history(config)`, pick the checkpoint where the graph was
paused at the HITL step, and resume *from that checkpoint* with a different
reviewer answer. LangGraph forks the thread there, so both outcomes stay in
the checkpoint history and can be compared side by side.

    uv run python exercises/bonus_1_time_travel.py --thread <id> --choice reject
    uv run python exercises/bonus_1_time_travel.py --thread <id> --choice edit --feedback "..."
    uv run python exercises/bonus_1_time_travel.py --thread <id> --answers "a1" "a2" ...   # escalation

Runs in dry-run mode (no GitHub comment) unless --post is given.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from rich.console import Console
from rich.table import Table

from common.db import db_path
from exercises.exercise_4_audit import build_graph


console = Console()
HITL_NODES = ("human_approval", "escalate")


def _summary(values: dict) -> str:
    a = values.get("analysis")
    return (
        f"final_action={values.get('final_action')}  "
        f"human_choice={values.get('human_choice')}  "
        f"confidence={a.confidence:.2f}  comments={len(a.comments)}" if a else "(no analysis)"
    )


async def run(thread_id: str, choice: str | None, feedback: str, answers: list[str] | None) -> None:
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        history = [s async for s in app.aget_state_history(cfg)]  # newest first
        if not history:
            console.print(f"[red]No checkpoints for thread {thread_id}[/red]")
            return

        table = Table(title=f"Checkpoint history — {thread_id}")
        for col in ("step", "checkpoint_id", "next", "interrupt"):
            table.add_column(col)
        for s in reversed(history):
            table.add_row(
                str(s.metadata.get("step")), s.config["configurable"]["checkpoint_id"][:13],
                ", ".join(s.next) or "END", "yes" if s.interrupts else "",
            )
        console.print(table)

        # Oldest checkpoint paused at a HITL node = the original decision point.
        paused = [s for s in reversed(history) if s.interrupts and set(s.next) & set(HITL_NODES)]
        if not paused:
            console.print("[red]This thread never paused at a HITL step — nothing to rewind to.[/red]")
            return
        fork_point = paused[0]
        kind = fork_point.interrupts[0].value["kind"]
        console.print(f"\nRewinding to step {fork_point.metadata.get('step')} "
                      f"(paused at [bold]{fork_point.next[0]}[/bold], kind={kind})")

        if kind == "approval_request":
            if not choice:
                raise SystemExit("--choice is required for an approval_request checkpoint")
            resume = {"choice": choice, "feedback": feedback}
        else:
            questions = fork_point.interrupts[0].value["questions"]
            given = answers or []
            resume = {q: (given[i] if i < len(given) else "No further context.")
                      for i, q in enumerate(questions)}

        original = history[0].values
        # Resuming fork_point.config directly would replay the *stored* resume
        # value (it is saved as a pending write on that checkpoint). Instead,
        # branch off with update_state — a fresh checkpoint whose parent is the
        # fork point, as if `route` had just finished — run it until it pauses
        # again, then resume with the new answer.
        fork_cfg = await app.aupdate_state(fork_point.config, None, as_node="route")
        paused_again = await app.ainvoke(None, fork_cfg)
        assert "__interrupt__" in paused_again, "fork did not pause at the HITL node"
        forked = await app.ainvoke(Command(resume=resume), cfg)

        console.rule("Compare outcomes")
        console.print(f"[dim]original:[/dim] {_summary(original)}")
        console.print(f"[bold]fork:    [/bold] {_summary(forked)}")
        n_after = len([s async for s in app.aget_state_history(cfg)])
        console.print(f"\n[dim]checkpoints in thread: {len(history)} → {n_after} "
                      f"(the fork is a new branch; the original branch is kept)[/dim]")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--thread", required=True)
    p.add_argument("--choice", choices=["approve", "reject", "edit"])
    p.add_argument("--feedback", default="")
    p.add_argument("--answers", nargs="*")
    p.add_argument("--post", action="store_true", help="Actually post to GitHub (default: dry run)")
    args = p.parse_args()
    if not args.post:
        os.environ["HITL_DRY_RUN"] = "1"
    asyncio.run(run(args.thread, args.choice, args.feedback, args.answers))


if __name__ == "__main__":
    main()
