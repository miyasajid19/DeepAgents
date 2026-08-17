from deepagents import create_deep_agent, DeepAgentState
from langchain.tools import  tool, ToolRuntime
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from rich import print
load_dotenv()

class MyCustomState(DeepAgentState):
    name: str
    age:int
    contact:str
    occupation:str
    hobbies:list[str]
    

@tool
def get_user_info(query: str, runtime: ToolRuntime[MyCustomState]) -> str:
    """Fetch user information from the custom state."""
    name = runtime.state["name"]
    age = runtime.state["age"]
    contact = runtime.state["contact"]
    occupation = runtime.state["occupation"]
    hobbies = runtime.state["hobbies"]

    return f"User Info:\nName: {name}\nAge: {age}\nContact: {contact}\nOccupation: {occupation}\nHobbies: {', '.join(hobbies)}"


model=ChatOpenAI(
    model_name="MiniMax-M3",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
)

agent=create_deep_agent(
    model=model,
    tools=[get_user_info],
    state_schema=MyCustomState,
)

result=agent.invoke(
    {
        "messages":[
            {
                "role": "user",
                "content": "Get my user information"
            }
        ],
        "name": "John Doe",
        "age": 30,
        "contact": "john.doe@example.com",
        "occupation": "Software Engineer",
        "hobbies": ["reading", "traveling", "coding"]
    }
)

print(result)