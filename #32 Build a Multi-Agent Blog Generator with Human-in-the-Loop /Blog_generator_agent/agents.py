import os
from langchain_groq import ChatGroq
from langchain_core import ChatPromptTemplet

#### Get LLm

def get_llm(model_name : str ="openai/gpt-oss-20b",temperature: float =0.5):
    api_key=os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model=model_name,temperature=temperature,api_key=api_key)
    return llm

### research agent

RESEARCH_PROMPT =ChatPromptTemplet.from_messages([
    {"role":"system","content":"""
     "You are a Research Agent. Given a blog topic and target audience, produce a clear, "
        "structured research outline. Include: \n"
        "1. 5-7 key points the blog should cover\n"
        "2. Important facts, stats, or examples for each point\n"
        "3. Suggested angle or hook\n"
        "Be concise. Use bullet points. Do NOT write the full blog yet."

        """},
    {"role":"user","content": "Topic:{topic},Audiance :{audiance},{revision_hints},Write the reaseach outline now. "}
])

def research_agent(llm:ChatGroq,topic:str,audiance:str,feedback:str="") -> str:
    revision_hints=f"human provided this feedback on your previous research - please address it:{feedback}"
    if not feedback:
        revision_hints = "thi is your first attempt."

    chain = RESEARCH_PROMPT | llm

    result = chain.invoke({
        "topic": topic,
        "audiance":audiance,
        "revision_hints":revision_hints,
    })

    return result.content

#writer agent

WRITER_PROMPT =ChatPromptTemplet.from_messages([
    {"role":"system","content":"""
     "You are a Blog Writer Agent. Using the research notes provided, write a complete, "
        "engaging blog post. \n"
        "Rules: \n"
        "- Length: 500-800 words \n"
        "- Structure: catchy title, intro hook, 3-5 sections with H2 headings, conclusion\n"
        "- Tone: clear, friendly, suited to the target audience\n"
        "- Use markdown formatting\n"
        "- Do NOT add a 'word count' line at the end"
        """},
    {"role":"user","content": """
    
    Topic:{topic},
    Audiance :{audiance},
    Research Notes: {research},
    {revision_hints},
     Write the full blog post now. """}
])

def writer_agent(llm:ChatGroq,topic:str,audiance:str,research :str="",feedback:str="") -> str:
    revision_hints=f"human provided this feedback on your previous draft and asked for these changes :{feedback},please apply these changes during writing the blog."
    if not feedback:
        revision_hints = "thi is your first attempt."

    chain = WRITER_PROMPT | llm

    result = chain.invoke({
        "topic": topic,
        "audiance":audiance,
        "research":research,
        "revision_hints":revision_hints,
    })

    return result.content

### Editior agent


EDITOR_PROMPT =ChatPromptTemplet.from_messages([
    {"role":"system","content":"""
     "You are an Editor Agent - the final quality gate before publishing.\n"
        "Take the draft and produce the FINAL polished version. Specifically: \n"|
        "- Fix grammar, spelling, and awkward phrasing\n"
        "- Tighten wordy sentences\n"
        "- Improve flow and transitions between sections\n"
        "- Make the title and intro more compelling if needed\n"
        "- Keep the same structure and markdown formatting\n"
        "- Blog workding should look like human written, not AI, and dont use any special chars and complex /fancy words.\n"
        "Output only the final polished blog post - no commentary."
        """},
    {"role":"user","content": """
    
    Topic:{topic},
    Draft : {draft}
    
    Return the Published Blog Post.
     """}
])

def editor_agent(llm:ChatGroq,topic:str,draft:str="") -> str:
    chain = EDITOR_PROMPT | llm

    result = chain.invoke({
        "topic": topic,
        "draft": draft,
    })

    return result.content