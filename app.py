"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Helpers ───────────────────────────────────────────────────────────────
RISK_ICON = {"low": "🟢", "med": "🟡", "high": "🔴"}
# MAX(risk_level) compares strings ('med' > 'low' > 'high'), so rank explicitly.
RISK_RANK_SQL = "CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END"


async def recent_threads(limit: int = 15) -> list[dict]:
    """Same query as `audit.replay.list_threads`, but returns rows instead of printing."""
    async with db_conn() as conn:
        async with conn.execute(
            f"""
            SELECT thread_id, pr_url,
                   MAX(timestamp) AS last_event,
                   CASE MAX({RISK_RANK_SQL}) WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END
                                  AS worst_risk,
                   COUNT(*)       AS events,
                   (SELECT action FROM audit_events e2
                     WHERE e2.thread_id = e.thread_id ORDER BY id DESC LIMIT 1) AS last_action
              FROM audit_events e
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def thread_state(thread_id: str):
    """Read the checkpointed graph state for a thread (no execution)."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        return await app.aget_state({"configurable": {"thread_id": thread_id}})


def load_thread(thread_id: str, pr_url: str) -> None:
    """Jump back into an existing session: its pending interrupt card or its final result."""
    snap = asyncio.run(thread_state(thread_id))
    st.session_state.thread_id = thread_id
    st.session_state.pr_url = pr_url
    st.session_state.interrupt_payload = snap.interrupts[0].value if snap.interrupts else None
    st.session_state.final = None if (snap.interrupts or snap.next) else dict(snap.values)


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    threads = asyncio.run(recent_threads())
    if not threads:
        st.caption("No sessions yet.")
    for t in threads:
        pr_label = t["pr_url"].split("github.com/")[-1]
        current = t["thread_id"] == st.session_state.thread_id
        if st.button(
            f"{RISK_ICON.get(t['worst_risk'], '⚪')} {pr_label} · {t['thread_id'][:8]}",
            key=f"thread_{t['thread_id']}",
            help=f"last event {t['last_event']} · {t['events']} events · last action: {t['last_action']}",
            type="primary" if current else "secondary",
            use_container_width=True,
        ):
            load_thread(t["thread_id"], t["pr_url"])
            st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", use_container_width=True):
        if not feedback.strip():
            st.warning("Write what should change in the Feedback box before choosing Edit.")
            return None
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("**Risk factors**\n" + "\n".join(f"- {r}" for r in payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        st.markdown("**The agent needs answers to these questions before it can finish the review:**")
        answers: dict[str, str] = {}
        for i, q in enumerate(payload["questions"]):
            answers[q] = st.text_area(f"{i + 1}. {q}", key=f"esc_q_{i}", height=80)
        if st.form_submit_button("Submit answers", type="primary"):
            if any(not a.strip() for a in answers.values()):
                st.warning("Please answer every question.")
                return None
            return {q: a.strip() for q, a in answers.items()}
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result
    st.rerun()  # refresh the sidebar so the new session shows up

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    analysis = final.get("analysis")
    if action.endswith("dry_run"):
        st.info(f"{action} — HITL_DRY_RUN=1, review not posted to GitHub")
    elif action.endswith("commit_failed"):
        st.error(f"{action} — posting the comment to GitHub failed (check GITHUB_TOKEN scope).")
    elif action.startswith("auto") or action.startswith("committed"):
        conf = f" — confidence {analysis.confidence:.0%}" if analysis else ""
        st.success(f"✓ {action}{conf} — comment posted to {st.session_state.pr_url}")
        st.link_button("View comment on GitHub", st.session_state.pr_url)
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")
    if analysis is not None:
        with st.expander("Final review", expanded=True):
            st.markdown(analysis.summary)
            for c in analysis.comments:
                st.markdown(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
