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
    raise ValueError(
        "MINIMAX_API_KEY is not set"
    )


if not MINIMAX_BASE_URL:
    raise ValueError(
        "MINIMAX_BASE_URL is not set"
    )


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
# 4. Get or Create Assistant-Scoped Sandbox
# ============================================================

def get_or_create_sandbox(
    assistant_id: str,
):

    sandbox_name = (
        f"deepagent-{assistant_id}"
    )

    print(
        f"\n[yellow]"
        f"Looking for Daytona sandbox:"
        f"[/yellow] {sandbox_name}"
    )


    # --------------------------------------------------------
    # Search existing sandboxes
    # --------------------------------------------------------

    for sandbox in daytona.list():

        if sandbox.name == sandbox_name:

            print(
                f"[green]"
                f"Existing sandbox found"
                f"[/green]"
            )

            print(
                f"Name: {sandbox.name}"
            )

            print(
                f"ID: {sandbox.id}"
            )

            return sandbox


    # --------------------------------------------------------
    # Create sandbox
    # --------------------------------------------------------

    print(
        f"[cyan]"
        f"Creating new Daytona sandbox"
        f"[/cyan]"
    )


    params = CreateSandboxFromSnapshotParams(
        name=sandbox_name,
    )


    sandbox = daytona.create(
        params
    )


    print(
        f"[green]"
        f"Sandbox created"
        f"[/green]"
    )

    print(
        f"Name: {sandbox.name}"
    )

    print(
        f"ID: {sandbox.id}"
    )


    return sandbox


# ============================================================
# 5. Agent Factory
# ============================================================

async def create_agent(
    config: RunnableConfig,
):

    assistant_id = (
        config[
            "configurable"
        ][
            "assistant_id"
        ]
    )


    print(
        f"\n[bold]"
        f"Assistant:"
        f"[/bold] {assistant_id}"
    )


    # --------------------------------------------------------
    # Assistant → Sandbox
    # --------------------------------------------------------

    sandbox = get_or_create_sandbox(
        assistant_id
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
# 6. Invoke Helper
# ============================================================

def invoke_agent(
    agent,
    message: str,
):

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ]
        }
    )


    print(
        "\n[bold blue]"
        "Agent:"
        "[/bold blue]"
    )


    print(
        result[
            "messages"
        ][
            -1
        ].content
    )


    return result


# ============================================================
# 7. Main
# ============================================================

async def main():

    ASSISTANT_ID = (
        "sajid-assistant-1"
    )


    # ========================================================
    # THREAD 1
    # ========================================================

    config_thread_1 = {

        "configurable": {

            "thread_id":
                "test-session-1",

            "assistant_id":
                ASSISTANT_ID,
        }
    }


    print(
        "\n"
        + "=" * 70
    )

    print(
        "THREAD 1"
    )

    print(
        "=" * 70
    )


    agent_1 = await create_agent(
        config_thread_1
    )


    # --------------------------------------------------------
    # Create project
    # --------------------------------------------------------

    invoke_agent(
        agent_1,
        """
Create a Python project in the sandbox.

Create a 3D scatter plot containing
1000 random points.

Requirements:

1. Generate 1000 random X values.
2. Generate 1000 random Y values.
3. Generate 1000 random Z values.
4. Create a 3D scatter plot using matplotlib.
5. Save it as scatter.png.
6. Create the Python source file.
7. Run the Python program.
8. Verify that scatter.png exists.
9. List all generated files.
""",
    )


    # ========================================================
    # THREAD 2 — SAME ASSISTANT
    # ========================================================

    config_thread_2 = {

        "configurable": {

            "thread_id":
                "testing",

            "assistant_id":
                ASSISTANT_ID,
        }
    }


    print(
        "\n"
        + "=" * 70
    )

    print(
        "THREAD 2 — SAME ASSISTANT"
    )

    print(
        "=" * 70
    )


    agent_2 = await create_agent(
        config_thread_2
    )


    invoke_agent(
        agent_2,
        """
List all files in the sandbox.

Check whether the project created
by the previous thread still exists.

Specifically check whether:

scatter.png

still exists.
""",
    )


    # ========================================================
    # THREAD 3 — SAME ASSISTANT
    # ========================================================

    config_thread_3 = {

        "configurable": {

            "thread_id":
                "another-thread",

            "assistant_id":
                ASSISTANT_ID,
        }
    }


    print(
        "\n"
        + "=" * 70
    )

    print(
        "THREAD 3 — SAME ASSISTANT"
    )

    print(
        "=" * 70
    )


    agent_3 = await create_agent(
        config_thread_3
    )


    invoke_agent(
        agent_3,
        """
List all files in the sandbox.

Tell me whether scatter.png
and the Python project are still present.
""",
    )


    # ========================================================
    # ASSISTANT 2
    # ========================================================

    SECOND_ASSISTANT_ID = (
        "another-assistant"
    )


    config_other_assistant = {

        "configurable": {

            "thread_id":
                "other-thread",

            "assistant_id":
                SECOND_ASSISTANT_ID,
        }
    }


    print(
        "\n"
        + "=" * 70
    )

    print(
        "ASSISTANT 2 — DIFFERENT SANDBOX"
    )

    print(
        "=" * 70
    )


    agent_4 = await create_agent(
        config_other_assistant
    )


    invoke_agent(
        agent_4,
        """
List all files in this sandbox.

Check whether scatter.png exists.
""",
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )