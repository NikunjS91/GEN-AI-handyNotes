from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from typing import Literal
import sqlite3

from state import BlogState
from agents import editor_agent, get_llm, writer_agent, research_agent

MAX_REVISION = 3
DB_PATH = "checkpoints.db"


def research_node(state: BlogState):
    llm = get_llm()
    research_data = research_agent(
        llm=llm,
        topic=state.topic,
        audience=state.audience,
        feedback=state.research_feedback,
    )
    state.research = research_data
    state.research_feedback = ""
    return state


def human_review_research_node(state: BlogState):
    decision = interrupt({
        "stage": "researcher_review",
        "research": state.research,
        "instructions": (
            "reply with 'approve' to continue to write",
            "or describe what to change to send it back to the researcher.",
        ),
    })
    if isinstance(decision, dict):
        action = decision.get("action", "approved")
        feedback = decision.get("feedback", "")
    else:
        text = str(decision)
        action = "approve" if text.lower() in ["approve", "approved", "okay", "ok", "done", "process", ""] else "revise"
        feedback = "" if action == "approve" else text

    state.research_feedback = feedback
    return state


def writer_node(state: BlogState):
    llm = get_llm()
    draft_data = writer_agent(
        llm=llm,
        topic=state.topic,
        audience=state.audience,
        research=state.research,
        feedback=state.draft_feedback,
    )
    state.draft = draft_data
    state.draft_feedback = ""
    state.revision_count += 1
    return state


def human_review_draft_node(state: BlogState):
    decision = interrupt({
        "stage": "draft_review",
        "draft": state.draft,
        "instructions": (
            "reply with 'approve' to continue to the editor",
            "or describe what to change to send it back to the writer.",
        ),
    })
    if isinstance(decision, dict):
        action = decision.get("action", "approved")
        feedback = decision.get("feedback", "")
    else:
        text = str(decision)
        action = "approve" if text.lower() in ["approve", "approved", "okay", "ok", "done", "process", ""] else "revise"
        feedback = "" if action == "approve" else text

    state.draft_feedback = feedback
    return state


def editor_node(state: BlogState):
    llm = get_llm()
    final = editor_agent(llm=llm, topic=state.topic, draft=state.draft)
    state.final_blog = final
    return state


#### conditional edges

def route_after_research_review(state: BlogState) -> Literal["research", "writer"]:
    if state.research_feedback:
        return "research"
    return "writer"


def route_after_draft_review(state: BlogState) -> Literal["writer", "edit"]:
    # Bug fix: was checking research_feedback instead of draft_feedback
    if state.draft_feedback and state.revision_count < MAX_REVISION:
        return "writer"
    return "edit"


### build and compile the graph

def build_blog_graph():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    builder = StateGraph(BlogState)

    builder.add_node("research", research_node)
    builder.add_node("review_researcher", human_review_research_node)
    builder.add_node("writer", writer_node)
    builder.add_node("review_draft", human_review_draft_node)
    builder.add_node("edit", editor_node)

    builder.add_edge(START, "research")
    builder.add_edge("research", "review_researcher")
    builder.add_conditional_edges(
        "review_researcher",
        route_after_research_review,
        {"research": "research", "writer": "writer"},
    )
    builder.add_edge("writer", "review_draft")
    builder.add_conditional_edges(
        "review_draft",
        route_after_draft_review,
        {"writer": "writer", "edit": "edit"},
    )
    builder.add_edge("edit", END)

    return builder.compile(checkpointer=checkpointer)
