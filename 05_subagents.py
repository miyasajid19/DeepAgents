from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()
from deepagents import CompiledSubAgent
from langchain.agents import create_agent


def internet_search(query: str) -> str:
    """Run a web search"""
    return f"Results for: {query}"

model= ChatOpenAI(
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

# coding_subagent = {
#     "name": "coding-agent",
#     "description": "Used to write and debug code.",
#     "system_prompt": "You are a coding expert.",
#     "tools": [],  # Add coding-related tools here
#     "model": model,  # Optional model override
# }
coding_agent_graph=create_agent(
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
    result = agent.invoke({"messages": [{"role": "user", "content": "Use the research subagent to find information about the latest advancements in AI. and spwan the coding subagent to write a simple Python script that prints 'Hello, World!' and then use the joke subagent to tell a joke about AI."}]})
    from rich import print
    print(result)
    print("[bold green]=================================================[/bold green]")
    print("[red] Agent's Response:[/red]")
    print(result["messages"][-1].content)