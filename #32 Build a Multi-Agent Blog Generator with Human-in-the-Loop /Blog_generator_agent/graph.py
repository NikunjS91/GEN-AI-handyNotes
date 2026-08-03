from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt,Command
from streamlit import feedback


from state import BlogState
from agents import editor_agent, get_llm,writer_agent,research_agent

MAX_REVISION = 3

def research_node(state:BlogState):
    """
    Researcher agent generates (or revises) the research outline
    """

    llm =get_llm()
    research_data = research_agent(
        llm=llm,
        topic =state.topic,
        audiance=state.audiance,
        feedback=state.research_feedback
    )
    state.research=research_data
    state.research_feedback=""

    return state
def human_review_research_node(state:BlogState):
    """pause and ask the human for approval or a feedback on research"""
    decision = interrupt({
        "stage":"researcher_review",
        "research":state.research,
         "instructions":(
             "reply with 'approve' to continue to write",
             "or describe what to change to send it back to the researcher."
         )
    })
    if isinstance(decision,dict):
        action=decision.get("action","approved")
        feedback=decision.get("feedback","")
    else:
        text =str(decision)
        action = "approve" if text.lower() in ["approve","approved","okay","ok","done","process",""] else "revise"
        feedback = "" if action =="approve" else text

    state.research_feedback=feedback
    return state

def writer_node(state:BlogState):
    pass


def editor_node(state:BlogState):
    pass