from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from deepagents.backends import FilesystemBackend
from rich import print
from dotenv import load_dotenv
import os
load_dotenv()

model=ChatOpenAI(
    model_name="MiniMax-M3",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
)



agent = create_deep_agent(
    model=model,
    backend=FilesystemBackend(root_dir=".",virtual_mode=True),
    memory=["PREFERENCES.md", "AGENTS.md"],
)



result=agent.invoke(
    {
        "messages":[
            {
                "role": "user",
                "content": "What do you know about me?"
            }
        ]
    }
)


print(result)


print("="*10)

result=agent.invoke(
    {
        "messages":[
            {
                "role": "user",
                "content": "give me a code for checking palindrome"
            }
        ]
    }
)



print(result)



print("="*10)

result=agent.invoke(
    {
        "messages":[
            {
                "role": "user",
                "content": "I have been learning deep agents and i want you save it"
            }
        ]
    }
)



print(result)



