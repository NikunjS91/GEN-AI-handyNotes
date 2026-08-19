import uuid
import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from langgraph.types import Command

from graph import build_blog_graph, MAX_REVISION
from state import BlogState

st.set_page_config(page_title="Blog Generator", layout="wide")

# ── graph is cached so the sqlite connection is reused across reruns ──────────
@st.cache_resource
def get_graph():
    return build_blog_graph()

graph = get_graph()


# ── helpers ───────────────────────────────────────────────────────────────────

STAGES = ["Research", "Review Research", "Writing", "Review Draft", "Editing", "Done"]

def make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}

def get_current_stage(thread_id: str) -> str:
    """Read the checkpointer to figure out where we are."""
    try:
        snapshot = graph.get_state(make_config(thread_id))
    except Exception:
        return "start"

    if not snapshot or not snapshot.values:
        return "start"

    values = snapshot.values
    next_nodes = list(snapshot.next) if snapshot.next else []

    if values.get("final_blog"):
        return "done"
    if "review_draft" in next_nodes or snapshot.tasks:
        # interrupted — check which interrupt it is
        for task in (snapshot.tasks or []):
            for interrupt_val in (task.interrupts or []):
                if isinstance(interrupt_val.value, dict):
                    return interrupt_val.value.get("stage", "unknown")
    if "edit" in next_nodes:
        return "editing"
    if "writer" in next_nodes:
        return "writing"
    if "review_researcher" in next_nodes:
        return "researching"
    return "unknown"

def read_state(thread_id: str) -> BlogState | None:
    try:
        snapshot = graph.get_state(make_config(thread_id))
        if snapshot and snapshot.values:
            return BlogState(**snapshot.values)
    except Exception:
        pass
    return None

def current_interrupt_stage(thread_id: str) -> str | None:
    """Return 'researcher_review' or 'draft_review' if paused at an interrupt, else None."""
    try:
        snapshot = graph.get_state(make_config(thread_id))
        for task in (snapshot.tasks or []):
            for interrupt_val in (task.interrupts or []):
                if isinstance(interrupt_val.value, dict):
                    return interrupt_val.value.get("stage")
    except Exception:
        pass
    return None


# ── sidebar progress bar ──────────────────────────────────────────────────────

def show_sidebar(stage_label: str):
    with st.sidebar:
        st.markdown("### Blog Pipeline")
        icons = {"done": "✅", "active": "🔄", "pending": "⬜"}
        steps = [
            ("Research", ["researching", "researcher_review"]),
            ("Review Research", ["researcher_review"]),
            ("Writing", ["writing", "draft_review"]),
            ("Review Draft", ["draft_review"]),
            ("Editing", ["editing"]),
            ("Done", ["done"]),
        ]
        passed = False
        for label, active_stages in steps:
            if stage_label in active_stages:
                st.markdown(f"**🔄 {label}**")
                passed = True
            elif not passed:
                st.markdown(f"✅ {label}")
            else:
                st.markdown(f"⬜ {label}")

        st.divider()
        if st.button("Start new post", use_container_width=True):
            st.query_params.clear()
            st.rerun()


# ── views ─────────────────────────────────────────────────────────────────────

def view_landing():
    show_sidebar("start")
    st.title("Blog Generator")
    st.markdown("Generate a polished blog post with AI — you review research and draft before publishing.")
    st.divider()

    with st.form("start_form"):
        topic = st.text_input("Blog topic", placeholder="e.g. How LLMs work")
        audience = st.text_input("Target audience", value="general", placeholder="e.g. developers, beginners")
        submitted = st.form_submit_button("Generate", type="primary")

    if submitted and topic:
        thread_id = str(uuid.uuid4())
        st.query_params["thread_id"] = thread_id
        config = make_config(thread_id)

        with st.spinner("Researching your topic..."):
            for _ in graph.stream(
                BlogState(topic=topic, audience=audience).model_dump(),
                config=config,
                stream_mode="values",
            ):
                pass

        st.rerun()


def view_research_review(thread_id: str, state: BlogState):
    show_sidebar("researcher_review")
    st.title("Review: Research Outline")
    st.caption(f"Topic: **{state.topic}** · Audience: **{state.audience}**")
    st.divider()

    st.markdown(state.research)
    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Approve & Write Draft", type="primary", use_container_width=True):
            config = make_config(thread_id)
            with st.spinner("Writing your draft..."):
                for _ in graph.stream(
                    Command(resume={"action": "approve", "feedback": ""}),
                    config=config,
                    stream_mode="values",
                ):
                    pass
            st.rerun()

    with col2:
        with st.expander("Request changes to research"):
            feedback = st.text_area("What should the researcher change?", key="research_fb")
            if st.button("Submit feedback", key="research_fb_submit"):
                if feedback.strip():
                    config = make_config(thread_id)
                    with st.spinner("Revising research..."):
                        for _ in graph.stream(
                            Command(resume={"action": "revise", "feedback": feedback}),
                            config=config,
                            stream_mode="values",
                        ):
                            pass
                    st.rerun()
                else:
                    st.warning("Please enter some feedback first.")


def view_draft_review(thread_id: str, state: BlogState):
    show_sidebar("draft_review")
    revision_num = state.revision_count
    st.title("Review: Blog Draft")
    st.caption(f"Topic: **{state.topic}** · Revision **{revision_num}** of {MAX_REVISION}")
    st.divider()

    st.markdown(state.draft)
    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Approve & Polish", type="primary", use_container_width=True):
            config = make_config(thread_id)
            with st.spinner("Editor is polishing your post..."):
                for _ in graph.stream(
                    Command(resume={"action": "approve", "feedback": ""}),
                    config=config,
                    stream_mode="values",
                ):
                    pass
            st.rerun()

    with col2:
        if revision_num < MAX_REVISION:
            with st.expander("Request changes to draft"):
                feedback = st.text_area("What should the writer change?", key="draft_fb")
                if st.button("Submit feedback", key="draft_fb_submit"):
                    if feedback.strip():
                        config = make_config(thread_id)
                        with st.spinner("Revising draft..."):
                            for _ in graph.stream(
                                Command(resume={"action": "revise", "feedback": feedback}),
                                config=config,
                                stream_mode="values",
                            ):
                                pass
                        st.rerun()
                    else:
                        st.warning("Please enter some feedback first.")
        else:
            st.info(f"Maximum revisions ({MAX_REVISION}) reached. Please approve to continue.")


def view_final(thread_id: str, state: BlogState):
    show_sidebar("done")
    st.title("Your Blog Post is Ready!")
    st.caption(f"Topic: **{state.topic}** · Audience: **{state.audience}**")
    st.divider()

    st.markdown(state.final_blog)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download as .md",
            data=state.final_blog,
            file_name=f"{state.topic[:40].replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        st.code(state.final_blog, language="markdown")


def view_processing(stage_label: str):
    show_sidebar(stage_label)
    st.title("Working...")
    st.spinner("The pipeline is running — refresh in a moment.")


# ── router ────────────────────────────────────────────────────────────────────

def main():
    thread_id = st.query_params.get("thread_id")

    if not thread_id:
        view_landing()
        return

    # Read current interrupt stage from checkpointer (reload-safe)
    interrupt_stage = current_interrupt_stage(thread_id)
    state = read_state(thread_id)

    if state is None:
        st.error("Could not find this session. Starting fresh.")
        st.query_params.clear()
        st.rerun()
        return

    if state.final_blog:
        view_final(thread_id, state)
    elif interrupt_stage == "researcher_review":
        view_research_review(thread_id, state)
    elif interrupt_stage == "draft_review":
        view_draft_review(thread_id, state)
    else:
        view_processing(interrupt_stage or "unknown")


main()
