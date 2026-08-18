from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)

load_dotenv()

model = ChatOpenAI(
    model="MiniMax-M3",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
)



agent=create_deep_agent(
    model=model,
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "Use the research subagent to find information about the latest advancements in AI. and spwan the coding subagent to write a simple Python script that prints 'Hello, World!' and then use the joke subagent to tell a joke about AI."}]})
    from rich import print
    print(result)
    print("[bold green]=================================================[/bold green]")
    print("[red] Agent's Response:[/red]")
    print(result["messages"][-1].content)
    print("[bold green]=================================================[/bold green]")
    
    
    # though it has no specialized agents but it will spilt task to general purpose subagent and will try to complete the task.