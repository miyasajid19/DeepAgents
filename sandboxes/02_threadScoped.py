import asyncio
import os

from dotenv import load_dotenv
from rich import print

from daytona import (
    Daytona,
    CreateSandboxFromSnapshotParams,
)

from deepagents import create_deep_agent
from langchain_core.runnables import RunnableConfig
from langchain_daytona import DaytonaSandbox
from langchain_openai import ChatOpenAI


# ============================================================
# 1. Environment
# ============================================================

load_dotenv()

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL")

if not MINIMAX_API_KEY:
    raise ValueError("MINIMAX_API_KEY is not set")

if not MINIMAX_BASE_URL:
    raise ValueError("MINIMAX_BASE_URL is not set")


# ============================================================
# 2. Daytona Client
# ============================================================

daytona = Daytona()


# ============================================================
# 3. Model
# ============================================================

model = ChatOpenAI(
    model="MiniMax-M3",
    base_url=MINIMAX_BASE_URL,
    api_key=MINIMAX_API_KEY,
    temperature=0.2,
)


# ============================================================
# 4. Get or Create Thread-Scoped Sandbox
# ============================================================

def get_or_create_sandbox(thread_id: str):

    sandbox_name = f"deepagent-{thread_id}"

    print(
        f"\n[yellow]Looking for Daytona sandbox:[/yellow] "
        f"{sandbox_name}"
    )

    # --------------------------------------------------------
    # Look for existing sandbox
    # --------------------------------------------------------

    for sandbox in daytona.list():

        if sandbox.name == sandbox_name:

            print(
                f"[green]Existing sandbox found:[/green] "
                f"{sandbox.name}"
            )

            print(
                f"[dim]Sandbox ID: {sandbox.id}[/dim]"
            )

            return sandbox


    # --------------------------------------------------------
    # Create new sandbox
    # --------------------------------------------------------

    print(
        f"[cyan]Creating Daytona sandbox:[/cyan] "
        f"{sandbox_name}"
    )

    params = CreateSandboxFromSnapshotParams(
        name=sandbox_name,
    )

    sandbox = daytona.create(params)

    print(
        f"[green]Sandbox created:[/green] "
        f"{sandbox.name}"
    )

    print(
        f"[dim]Sandbox ID: {sandbox.id}[/dim]"
    )

    return sandbox


# ============================================================
# 5. Agent Factory
# ============================================================

async def agent(config: RunnableConfig):

    thread_id = config["configurable"]["thread_id"]

    print(
        f"\n[bold]Thread:[/bold] {thread_id}"
    )

    # --------------------------------------------------------
    # Thread → Daytona Sandbox
    # --------------------------------------------------------

    sandbox = get_or_create_sandbox(
        thread_id
    )

    # --------------------------------------------------------
    # Daytona Backend
    # --------------------------------------------------------

    backend = DaytonaSandbox(
        sandbox=sandbox
    )

    # --------------------------------------------------------
    # Deep Agent
    # --------------------------------------------------------

    return create_deep_agent(
        model=model,
        backend=backend,
    )


# ============================================================
# 6. Main
# ============================================================

async def main():

    THREAD_ID = "test-session-1"

    config = {
        "configurable": {
            "thread_id": THREAD_ID
        }
    }


    # ========================================================
    # Create / Reuse Agent
    # ========================================================

    compiled_agent = await agent(
        config
    )


    # ========================================================
    # TASK 1
    # ========================================================

    print(
        "\n[bold green]"
        "TASK 1: Create Python project"
        "[/bold green]"
    )

    result = compiled_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": """
Create a Python project in the sandbox.

Requirements:
create a scattor plat of 1000 random points in 2D space and save it as a PNG file.
""",
                }
            ]
        }
    )


    print(
        "\n[bold blue]Agent response:[/bold blue]"
    )

    print(
        result["messages"][-1].content
    )


    # ========================================================
    # TASK 2
    # ========================================================

    print(
        "\n[bold green]"
        "TASK 2: Inspect sandbox"
        "[/bold green]"
    )

    result = compiled_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": """
List every file and directory currently present
in the sandbox.

Show the complete directory tree.
""",
                }
            ]
        }
    )


    print(
        "\n[bold blue]Agent response:[/bold blue]"
    )

    print(
        result["messages"][-1].content
    )


    # ========================================================
    # TASK 3
    # ========================================================

    print(
        "\n[bold green]"
        "TASK 3: Verify persistence"
        "[/bold green]"
    )

    print(
        "\nCreating the agent again using the SAME thread..."
    )
    
    compiled_agent_2 = await agent(
        config
    )


    result = compiled_agent_2.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": """
Check whether the Python project created earlier
still exists.

List the files and tell me whether the previous
project is still present.
""",
                }
            ]
        }
    )


    print(
        "\n[bold blue]Agent response:[/bold blue]"
    )

    print(
        result["messages"][-1].content
    )



    # =======================================================
    # Task 4: check sandbox persistence across threads
    # =======================================================
    config = {
        "configurable": {
            "thread_id": "testing"
        }
    }
    
    agent_3 = await agent(
        config
    )
    result= agent_3.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": """
Check whether the Python project created earlier
still exists."""
                }
            ]
        }
    )
    print(
        "\n[bold blue]Agent response:[/bold blue]"
    )

    print(
        result["messages"][-1].content
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )