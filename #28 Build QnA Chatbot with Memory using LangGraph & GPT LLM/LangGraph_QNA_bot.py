from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from pydantic import BaseModel
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated


class ChatState(BaseModel):
    messages : Annotated[list,add_messages]


llm = ChatGroq(model="openai/gpt-oss-20b")

def chatBotNode(state:ChatState) -> ChatState:
    res = llm.invoke(state.messages)
    state.messages =[res]
    return state

memory = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("ChatBot",chatBotNode)

graph.add_edge(START,"ChatBot")
graph.add_edge("ChatBot",END)

graph=graph.compile(checkpointer=memory)

config ={"configurable":{"thread_id":"my-bot-1"}}

while True:
    query =input("User: ")
    if query.lower() in ["quit", "exit","stop"]:
        print("bye bye")
        break 

    config ={"configurable":{"thread_id":"my-bot-1"}}
    res = graph.invoke({"messages":[{"role":"user","content":query}]},
                        config
                    )
    ans = res["messages"][-1].content
    print("AI:" ,ans)