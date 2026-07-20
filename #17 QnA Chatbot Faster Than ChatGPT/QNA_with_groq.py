from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st 
from langchain_community.utilities import GoogleSerperAPIWrapper


llm = ChatGroq(model="openai/gpt-oss-120b",streaming=True)
search = GoogleSerperAPIWrapper()
tool=[search.run]

if  "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history =[]

agent = create_agent(
    model=llm,
    tools= tool,
    checkpointer=st.session_state.memory,
    system_prompt="you are a amazing ai agent who uses google as tools to search for answers."
)


st.subheader("Fastest QNA bot with WebSearch Integreation.")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Can ask me a question? ")


if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content":"query"})

    response = agent.stream({"messages":[{"role":"user","content":query}]},
                {"configurable":{"thread_id":"1"}},
                stream_mode="messages"
                )
    ai_container =st.chat_message("ai")
    with ai_container:          #referance created
        space = st.empty()      #blank space created

        message =""             #in this variable the chunk messages will be stored.

        for chunk in response:
            message = message + chunk[0].content        #based on the message comming from response it will be appended to message varibale.
            space.write(message)                            #writing in that space over front end.
        st.session_state.history.append({"role":"ai","content":message})
    #ans = response["messages"][-1].content
    #st.chat_message("ai").markdown(ans)
    