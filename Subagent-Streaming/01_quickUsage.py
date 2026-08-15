from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tavily import TavilyClient
import re
import os

load_dotenv()
from deepagents import CompiledSubAgent
from langchain.agents import create_agent


def internet_search(query: str) -> str:
    """Run a web search"""
    if not os.getenv("TAVILY_API_KEY"):
        return "TAVILY_API_KEY is missing. Unable to run live web search."

    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily_client.search(query=query, max_results=5)
    results = response.get("results", [])

    if not results:
        return "No web results returned."

    lines = []
    for idx, item in enumerate(results, start=1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = item.get("content", "")
        lines.append(f"{idx}. {title}\nURL: {url}\n{content}")

    return "\n\n".join(lines)


def strip_think_blocks(text: str) -> str:
    """Remove chain-of-thought style blocks and extra whitespace from streamed text."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

model = ChatOpenAI(
    model_name="MiniMax-M3",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
)
research_subagent = {
    "name": "research-agent",
    "description": "Used to research in-depth questions using web search.",
    "system_prompt": "You are a thorough research specialist.",
    "tools": [internet_search],
    "model": model,  # Optional model override
}


coding_agent_graph = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a coding expert.",
)

coding_subagent = CompiledSubAgent(
    name="coding-agent",
    description="Used to write and debug code.",
    runnable=coding_agent_graph,
)

joke_subagent = {
    "name": "joke-agent",
    "description": "Used to tell jokes and entertain.",
    "system_prompt": "You are a comedian.",
    "tools": [],  # Add joke-related tools here
    "model": model,  # Optional model override
}
agent = create_deep_agent(
    model=model,
    subagents=[research_subagent, coding_subagent, joke_subagent],
)

if __name__ == "__main__":
    stream = agent.stream_events(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Use the research subagent to find information about Apple, then spawn the coding subagent to write a simple Python script that prints 'Apple!', and then use the joke subagent to tell a joke about Apple."
                }
            ]
        },
        version="v3"
    )
    from rich import print

    subagents_name: list[str] = []
    for subagent in stream.subagents:
        print(subagent.name, subagent.path, subagent.status)

        for message in subagent.messages:
            raw_text = getattr(message, "text", "")
            text = raw_text if isinstance(raw_text, str) else str(raw_text)
            visible_text = strip_think_blocks(text)
            if visible_text:
                print(visible_text)

        subagents_name.append(subagent.name)
        print("another subagent invoked\n")

    print(subagents_name)