from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

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
    content="""You are an expert researcher. Your job is to conduct thorough research and then write a polished report."""
)

agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    system_prompt=SYSTEM_PROMPT,
    backend=FilesystemBackend(root_dir=str(Path(__file__).parent), virtual_mode=True),
    memory=["/AGENTS.md"],
    skills=["/skills"],
)

result = agent.invoke({"messages": [HumanMessage(content="use your skill and memory and summarize it")]})

from rich import print
print(result)
print("[bold green]=================================================[/bold green]")
print("[red] Agent's Response:[/red]")
print(result["messages"][-1].content)