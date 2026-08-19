from deepagents import AsyncSubAgent, create_deep_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

model = ChatOpenAI(
    model_name="Minimax-M3",
    api_key=os.environ.get("MINIMAX_API_KEY"),
    base_url=os.environ.get("MINIMAX_BASE_URL"),
)

submodel = ChatOpenAI(
    model_name="Minimax-M2.7",
    api_key=os.environ.get("MINIMAX_API_KEY"),
    base_url=os.environ.get("MINIMAX_BASE_URL"),
)

async_subagents = [
    AsyncSubAgent(
        name="researcher",
        description="Research agent for information gathering and synthesis",
        graph_id="researcher",
        url="http://localhost:8123",
    ),
    AsyncSubAgent(
        name="coder",
        description="Coding agent for code generation and review",
        graph_id="coder",
        url="http://127.0.0.1:2024/",
    ),
]

agent = create_deep_agent(
    model="google_genai:gemini-3.6-flash",
    subagents=async_subagents,
)


async def main():
    response = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": (
                "Research the latest AI trends using the researcher "
                "subagent, and then generate a code snippet related "
                "to those trends using the coder subagent."
            )
        }]
    })

    print(response)


if __name__ == "__main__":
    asyncio.run(main())