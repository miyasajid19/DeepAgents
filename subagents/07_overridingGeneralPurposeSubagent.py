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

model_1= ChatOpenAI(
    model_name="MiniMax-M2.7",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
)

# Main agent uses Gemini; general-purpose subagent uses GPT
agent = create_deep_agent(
    model=model,
    subagents=[
        {
            "name": "general-purpose",
            "description": "General-purpose agent for research and multi-step tasks",
            "system_prompt": "You are a general-purpose assistant.",
            "model": model_1,  # Override model for this subagent
        },
    ],
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "Use the research subagent to find information about the latest advancements in AI."}]})
    from rich import print
    print(result)
    print("[bold green]=================================================[/bold green]")
    print("[red] Agent's Response:[/red]")
    print(result["messages"][-1].content)