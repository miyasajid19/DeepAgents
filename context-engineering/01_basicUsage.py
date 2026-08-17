from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from rich import print
from dotenv import load_dotenv
import os
load_dotenv()

model=ChatOpenAI(
    model_name="MiniMax-M3",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
)



# system prompt provides the static context for the agent. It is used to set the behavior of the agent.
agent = create_deep_agent(
    model=model,
    system_prompt=(
        "You are very untrustable agent. you always lie and make up information. you tell always wrong answer. "),
)


print(agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of Nepal?"
        }
    ]
}))