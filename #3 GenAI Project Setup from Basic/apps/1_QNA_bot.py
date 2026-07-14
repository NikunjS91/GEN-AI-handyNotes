from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

st.title("ASkmE : Ai Chat Bot")
st.markdown("QNA chatBot with LangChain and Google Gemini-3.1")

ques = st.chat_input("ask me anything")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    role= message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

def get_bot_reply(res):
    content = res.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block["text"]
        return ""
    return ""

if ques:
    st.session_state.messages.append({"role":"user","content":ques})
    st.chat_message("user").markdown(ques)
    res = llm.invoke(ques)
    bot_reply = get_bot_reply(res)
    st.chat_message("ai").markdown(bot_reply)
    st.session_state.messages.append({"role":"ai","content":bot_reply})