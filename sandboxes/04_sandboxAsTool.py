import os

from dotenv import load_dotenv
from rich import print

from daytona import Daytona
from deepagents import create_deep_agent
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

client = Daytona()


# ============================================================
# 3. Create Sandbox
# ============================================================

print("[yellow]Creating Daytona sandbox...[/yellow]")

sandbox = client.create()

print(
    f"[green]Sandbox created:[/green] {sandbox.id}"
)


# ============================================================
# 4. Daytona Backend
# ============================================================

backend = DaytonaSandbox(
    sandbox=sandbox,
)


# ============================================================
# 5. Model
# ============================================================

model = ChatOpenAI(
    model="MiniMax-M3",
    base_url=MINIMAX_BASE_URL,
    api_key=MINIMAX_API_KEY,
    temperature=0.2,
)


# ============================================================
# 6. Agent
# ============================================================

agent = create_deep_agent(
    model=model,
    backend=backend,
    system_prompt="""
You are a coding assistant with access to a Daytona sandbox.

IMPORTANT:

The sandbox working directory is:

/home/daytona

Always create project files inside:

/home/daytona

Never create files directly under:

/

For example, use:

/home/daytona/hello.py

instead of:

/hello.py

You can:

- create files
- read files
- modify files
- delete files
- create directories
- execute shell commands
- execute Python programs
- inspect the filesystem

When asked to create or execute something, actually perform
the operation inside the Daytona sandbox.
""",
)


# ============================================================
# 7. Run Agent
# ============================================================

try:

    print(
        "\n[bold green]"
        "Invoking Deep Agent..."
        "[/bold green]"
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": """
Create a Python file at:

/home/daytona/hello.py

The file must contain:

print("Hello World")

Then:

1. Run the Python file.
2. Verify the output.
3. List the files inside /home/daytona.
4. Tell me the exact location of hello.py.
""",
                }
            ]
        }
    )

    print(
        "\n[bold blue]"
        "Agent response:"
        "[/bold blue]"
    )

    print(
        result["messages"][-1].content
    )


except Exception as e:

    print(
        "\n[bold red]"
        "Agent failed:"
        "[/bold red]"
    )

    print(
        f"{type(e).__name__}: {e}"
    )


finally:

    # ========================================================
    # Cleanup
    # ========================================================

    print(
        "\n[yellow]"
        "Deleting Daytona sandbox..."
        "[/yellow]"
    )

    client.delete(
        sandbox,
        wait=True,
    )

    print(
        "[green]"
        "Sandbox deleted successfully."
        "[/green]"
    )