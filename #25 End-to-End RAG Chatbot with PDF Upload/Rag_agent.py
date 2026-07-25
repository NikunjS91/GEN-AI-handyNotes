from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def process_document(path):

    #load the document
    loader = PyPDFDirectoryLoader(path)
    document =loader.load()

    ##split into cunks
    splitter = RecursiveCharacterTextSplitter(chunk_size =800,chunk_overlap =200)
    splitted_text = splitter.split_documents(documents=document)


    ##vector and embedding
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_stored = InMemoryVectorStore.from_documents(
        documents=splitted_text,
        embedding=embedding,
        
    )

    @tool
    def retriver_tool(query:str):
        """
            this tool can help you find relevant infromation from the document provided or we can say a knowledge base.
        """
        docs = vector_stored.similarity_search(query=query)
        context =""

        for doc in docs:
            context = doc.page_content + "\n\n"
        return context

    llm = ChatOpenAI(model="gpt-4o")


    System_prompt= """
            you are a helpful assistant that answer questions using the retrived context.
            ALWAYS use the 'retrivever_tool' tool for question requiring external knowledge.
    """

    memeory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retriver_tool],
        system_prompt=System_prompt,
        checkpointer=memeory,
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True
    



###upload document
if not st.session_state.document_uploaded:
    uploader = st.file_uploader(label="Please Upload PDF File",type=["pdf"],accept_multiple_files=True)
    if uploader:
         with st.spinner("Uploading Please wait"):
            path = "./Uploaded_PDF_Files/"
            for file in uploader:
                with open(path + file.name,"wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            st.rerun()

#chat Ui

if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content") 
        st.chat_message(role).markdown(content)

    query = st.chat_input("ask me anything based on the provided docuemnts...")
    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role":"user","content":query})
        response=st.session_state.agent.invoke(
            {"messages":[{"role":"user","content":query}]},
            {"configurable":{"thread_id":1}}
        )

        answer= response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai","content":answer})