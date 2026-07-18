from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


search = GoogleSerperAPIWrapper()
model =ChatOpenAI(model="gpt-4o")

agent = create_agent(
    model=model,
    tools=[search.run],
    system_prompt="you are a agent who search google for answer based on the question."
)


while True:
    query = input("User : ")
    if query.lower() == "exit":
        break
    response = agent.invoke({"messages":[{"role":"user","content":query}]})
    print("ai_system: ",response["messages"][-1].content)