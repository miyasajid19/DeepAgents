import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich import print

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

load_dotenv()

model = ChatOpenAI(
    model_name="MiniMax-M3",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
)

agent = create_deep_agent(
    model=model,
    backend=FilesystemBackend(root_dir=".", virtual_mode=True),
    skills=["./skills/"],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What do you know about me?",
            }
        ]
    }
)

print(result)

print("=" * 10)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "give me a code for checking palindrome",
            }
        ]
    }
)

print(result)
