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
model_1 = ChatOpenAI(
    model="MiniMax-M2.7",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
)
coding_subagent = {
    "name": "coding-agent",
    "description": "Used to write and debug code.",
    "system_prompt": "You are a coding expert in C.",
}

joke_subagent = {
    "name": "joke-agent",
    "description": "Used to tell jokes and entertain.",
    "system_prompt": "You are a comedian.",
    "tools": [],  # Add joke-related tools here
    "model": model_1,  # Optional model override
}


register_harness_profile(
    "openai:MiniMax-M3",
    HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),

    ),
)

agent=create_deep_agent(
    model=model,
    middleware=[],
)

agent_1=create_deep_agent(
    model=model,
    subagents=[coding_subagent, joke_subagent],
)
if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "Use the research subagent to find information about the latest advancements in AI. and spwan the coding subagent to write a simple Python script that prints 'Hello, World!' and then use the joke subagent to tell a joke about AI."}]})
    from rich import print
    print(result)
    print("[bold green]=================================================[/bold green]")
    print("[red] Agent's Response:[/red]")
    print(result["messages"][-1].content)
    print("[bold green]=================================================[/bold green]")
    
    result=agent_1.invoke({"messages": [{"role": "user", "content": "Use the research subagent to find information about the latest advancements in AI. and spwan the coding subagent to write a simple Python script that prints 'Hello, World!' and then use the joke subagent to tell a joke about AI."}]})
    print("[bold green]=================================================[/bold green]")
    print(result)
    print("[bold green]=================================================[/bold green]")
    print("[red] Agent's Response:[/red]")
    print(result["messages"][-1].content)
    print("[bold green]=================================================[/bold green]")
    
# when enabled the supervisor will spilt the task to general purpose subagent and the subagents will be used to complete the task.
# whe disabled the supervisor will not spilt the task to general purpose subagent and it will work as a normal agent without subagents.
# but if you give subagents the it will use the subagents to complete the task.