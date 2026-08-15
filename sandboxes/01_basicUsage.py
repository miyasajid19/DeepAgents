from daytona import Daytona
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain_daytona import DaytonaSandbox
from rich import print
from dotenv import load_dotenv
import os

load_dotenv()

DAYTONA_API_KEY = os.getenv("DAYTONA_API_KEY")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_API_URL = os.getenv("MINIMAX_BASE_URL")

if not DAYTONA_API_KEY:
    raise ValueError("DAYTONA_API_KEY is missing")

if not MINIMAX_API_KEY:
    raise ValueError("MINIMAX_API_KEY is missing")

if not MINIMAX_API_URL:
    raise ValueError("MINIMAX_API_URL is missing")

sandbox = Daytona().create()
backend = DaytonaSandbox(sandbox=sandbox)

agent = create_deep_agent(
    model=ChatOpenAI(
        model="MiniMax-M3",
        base_url=MINIMAX_API_URL,
        api_key=MINIMAX_API_KEY,
    ),
    system_prompt="You are a Python coding assistant with sandbox access.",
    backend=backend,
)

try:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        """Create a Python project in the sandbox that:
1. Creates a CSV containing 100 random students.
2. Calculates their average marks.
3. Identifies the top 10 students.
4. Creates a matplotlib visualization.
5. Saves the chart as PNG.
6. Writes a README explaining the project.
7. Run the program and verify that everything works."""
                    ),
                }
            ]
        }
    )

    print(result)

finally:
    sandbox.stop()