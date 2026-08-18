from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()
from deepagents import CompiledSubAgent
from langchain.agents import create_agent


model= ChatOpenAI(
    model_name="MiniMax-M3",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
)
coding_agent_graph=create_agent(
    model=model,
    tools=[],
    system_prompt="You are a coding expert.",
)

coding_subagent = CompiledSubAgent(
    name="coding-agent",
    description="Used to write and debug code. You are a coding expert in C. when you are told to code in C or any C code the  this agent must be invoked ",
    runnable=coding_agent_graph,
)
agent = create_deep_agent(
    model=model,
    subagents=[coding_subagent],
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "Write a simple Python script that prints 'Hello, World!'"}]})
    from rich import print
    print(result)
    print("[bold green]=================================================[/bold green]")
    print("[red] Agent's Response:[/red]")
    print(result["messages"][-1].content)
    print("[bold green]=================================================[/bold green]")
    print("[bold green]=================================================[/bold green]")
    result = agent.invoke({"messages": [{"role": "user", "content": "Write a simple C  script that prints 'Hello, World!'"}]})
    from rich import print
    print(result)
    print("[bold green]=================================================[/bold green]")
    print("[red] Agent's Response:[/red]")
    print(result["messages"][-1].content)