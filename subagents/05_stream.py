from deepagents import (
    create_deep_agent
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from rich import print
load_dotenv()


agent = create_deep_agent(
    model=ChatOpenAI(
        model_name="MiniMax-M3",
        api_key=os.getenv("MINIMAX_API_KEY"),   
        base_url=os.getenv("MINIMAX_BASE_URL")),
    system_prompt=(
        "You are a project coordinator with no research knowledge. "
        "For every user request, you must call the task() tool with "
        "subagent_type set to research-agent,subagent_type set to coding-agent,subagent_type set to Joke-agent. Never answer research "
        "questions yourself."
    ),
    subagents=[
        {
            "name": "research-agent",
            "description": (
                "Delegate research to this subagent. Give one topic at a time."
            ),
            "system_prompt": (
                "You are a great researcher. Return a brief summary."
            ),
        },
        {
            "name": "coding-agent",
            "description": (
                "Delegate coding tasks to this subagent. Provide clear instructions and requirements."
            ),
            "system_prompt": (
                "You are a skilled programmer. Write clean, efficient code."
            ),
        },
        {
            "name": "Joke-agent",
            "description": (
                "Delegate joke-telling tasks to this subagent. Provide clear instructions and requirements."
            ),
            "system_prompt": (
                "You are a comedian. Tell jokes in a funny and engaging way."
            ),
        },
    ],
    name="main-agent",
)

if __name__ == "__main__":
    stream = agent.stream_events(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Research one recent advance in quantum computing. and tell me joke on AI. and can you give me python code for reversing string",
                }
            ]
        },
        version="v3",
    )

    coordinator_messages: list[str] = []
    subagent_handles = []

    for name, item in stream.interleave("messages", "subagents"):
        if name == "messages":
            print("[coordinator]", item.text)
            coordinator_messages.append(item.text)
        else:
            print(f"[{item.name}] started")
            subagent_handles.append(item)
            for message in item.messages:
                print(f"[{item.name}]", message.text)
            print(f"[{item.name}] status: {item.status}")