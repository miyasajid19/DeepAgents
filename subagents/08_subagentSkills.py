# if main agent has skills then subagents will automatically inherit those skills.

# giving skills to subagent

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()
from deepagents.backends import FilesystemBackend

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

sajid_subagent={
    "name":"temp-agent",
    "description":"This is a temp agent having skills about sajid miya. Invoke this when you want to know about sajid miya.",
    "system_prompt":"You are a temp agent having skills about sajid miya. Invoke this when you want to know about sajid miya. and always tell what skills set you have and tell from what what skills you are returning about sajid",
}
sajid_subagent_skills={
    "name":"temp-agent",
    "description":"This is a temp agent having skills about sajid miya. Invoke this when you want to know about sajid miya.",
    "system_prompt":"You are a temp agent having skills about sajid miya. Invoke this when you want to know about sajid miya. and always tell what skills set you have and tell from what what skills you are returning about sajid",
}

agent=create_deep_agent(
    model=model,
    backend=FilesystemBackend(root_dir='.',virtual_mode=True),
    skills=['skills/ponytail/','skills/sajid/'],
    subagents=[sajid_subagent],
)

agent_1=create_deep_agent(
    model=model,
    backend=FilesystemBackend(root_dir='.',virtual_mode=True),
    subagents=[sajid_subagent_skills],
)
if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "Tell me about sajid miya. and tell me what skills you have and tell from what what skills you are returning about sajid"}]})
    from rich import print
    print(result)
    print("[bold green]=================================================[/bold green]")
    print("[red] Agent's Response:[/red]")
    print(result["messages"][-1].content)
    print("[bold green]=================================================[/bold green]")
    print("[bold green]=================================================[/bold green]")
    result = agent_1.invoke({"messages": [{"role": "user", "content": "Tell me about sajid miya. and tell me what skills you have and tell from what what skills you are returning about sajid"}]})
    from rich import print
    print(result)
    print("[bold green]=================================================[/bold green]")
    print("[red] Agent's Response:[/red]")
    print(result["messages"][-1].content)