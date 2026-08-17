from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, FilesystemBackend
from langgraph.store.memory import InMemoryStore
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from rich import print
load_dotenv()

model=ChatOpenAI(
    model_name="MiniMax-M3",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
)

store = InMemoryStore()

agent = create_deep_agent(
    model=model,
    store=store,
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": FilesystemBackend(root_dir='./memories/',virtual_mode=True),
        },
    ),
    system_prompt="""When users tell you their preferences, save them to
    memories/preferences.txt so you remember them in future conversations.""",
)

result=agent.invoke(
    {
        "messages":[
            {
                "role": "user",
                "content": "what is my name and my punch line?"
            }
        ]
        
    }
)

print(result)

print("=="*10)

result=agent.invoke(
    {
        "messages":[
            {
                "role": "user",
                "content": "user have experience with crewai and n8n add this in prefrences too"
            }
        ]
        
    }
)

print(result)
