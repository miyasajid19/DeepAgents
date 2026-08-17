import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich import print
from langchain.tools import tool
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

load_dotenv()

model = ChatOpenAI(
    model_name="MiniMax-M3",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
)



@tool(parse_docstring=True)
def sajid() -> str:
    '''
        This is the tool that gives agent about the information about sajid. It is a simple tool that returns the information about sajid.
    '''
    return "Sajid is a software engineer who is passionate about building intelligent systems. He has experience in developing AI and machine learning applications, and he enjoys exploring new technologies and frameworks. In his free time, he likes to read about the latest advancements in AI and contribute to open-source projects. He is 21 years old and currently pursuing his studies in computer science and engineering in Thapar Institute of Engineering and Technology, Patiala. He is originally from Gorkha,Nepal. He had completed his schooling from The Old Capital Secondary School, Raniban, Gorkha. He completed his high school from St. Xavier's College, Maitighar, Kathmandu. He is currently pursuing his undergraduate studies in computer science and engineering at Thapar Institute of Engineering and Technology, Patiala."


agent=create_deep_agent(
    model=model,
    tools=[sajid],
)

result=agent.invoke(
    {
        "messages":[
            {
                "role": "user",
                "content": "What do you know about sajid?"
            }
        ]
    }
)

print(result)