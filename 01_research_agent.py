from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from deepagents import create_deep_agent
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()  # Load environment variables from .env file
model = ChatOpenAI(
    model_name="MiniMax-M3",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
)


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    
SYSTEM_PROMPT = SystemMessage(
    content="""You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
""")


agent=create_deep_agent(
    model=model,
    tools=[internet_search],
    system_prompt=SYSTEM_PROMPT,
)


result = agent.invoke({"messages": [{"role": "user", "content": "can you research on prime minister of nepal and how he become the PM"}]})

# Print the agent's response
print(result["messages"][-1].content)