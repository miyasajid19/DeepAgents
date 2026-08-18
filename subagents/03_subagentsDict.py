from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()


model= ChatOpenAI(
    model_name="MiniMax-M3",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
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
    subagents=[joke_subagent],
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "tell a joke about AI."}]})
    from rich import print
    print(result)
    print("[bold green]=================================================[/bold green]")
    print("[red] Agent's Response:[/red]")
    print(result["messages"][-1].content)