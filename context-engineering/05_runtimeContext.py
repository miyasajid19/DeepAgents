import os
from rich import print
from dotenv import load_dotenv
from langchain.tools import tool
from dataclasses import dataclass
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain.tools import ToolRuntime, tool

load_dotenv()

@dataclass
class Context:
    user_id: str
    user_age:int
    phone_no:str
    email:str


@tool
def fetch_user_data(query: str, runtime: ToolRuntime[Context]) -> str:
    """Fetch data for the current user."""
    print("="*10)
    print(runtime)
    print("="*10)
    user_id = runtime.context.user_id
    return f"Data for user {user_id}: {query}"

model = ChatOpenAI(
    model_name="MiniMax-M3",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
)

agent = create_deep_agent(
    model=model,
    tools=[fetch_user_data],
    context_schema=Context,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Get my data"}]},
    context=Context(user_id="user-123", user_age=21, phone_no="123-456-7890", email="user@example.com"),
)

print(result)