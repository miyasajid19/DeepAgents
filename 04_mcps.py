import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()
from rich import print
async def main():
    client = MultiServerMCPClient(
        {
            "my_server": {
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp",
            }
        }
    )

    # Get tools from the MCP server
    tools = await client.get_tools()

    print("Available MCP tools:")
    for tool in tools:
        print(f"- {tool}")

    # Create Deep Agent
    model = ChatOpenAI(
        model_name="MiniMax-M3",
        api_key=os.getenv("MINIMAX_API_KEY"),
        base_url=os.getenv("MINIMAX_BASE_URL"),
    )
    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt="If I have Tool then i have to use the tool strictly.",
    )

    # Ask the agent to use the MCP tools
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the sum and product of 10 and 3?",
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": "1"
            }
        },
    )

    print("\nAgent result:")
    print(result)
    
    print("[bold green]=================================================[/bold green]")
    print("[red] Agent's Response:[/red]")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())