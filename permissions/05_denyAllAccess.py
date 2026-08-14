import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from deepagents import create_deep_agent, FilesystemPermission
from deepagents.backends import (
    CompositeBackend,
    StateBackend,
    FilesystemBackend,
)


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
# 2. Model
# ============================================================

model = ChatOpenAI(
    model="MiniMax-M3",
    base_url=MINIMAX_BASE_URL,
    api_key=MINIMAX_API_KEY,
    temperature=0.2,
)


# ============================================================
# 3. Checkpointer
# ============================================================

"""
InMemorySaver stores LangGraph checkpoint/state information
in RAM.

It is lost when the Python process exits.
"""

checkpointer = InMemorySaver()


# ============================================================
# 4. Physical directories
# ============================================================

WORKSPACE = Path("workspace").resolve()

MEMORIES_DIR = WORKSPACE / "memories"
POLICIES_DIR = WORKSPACE / "policies"

WORKSPACE.mkdir(
    parents=True,
    exist_ok=True,
)

MEMORIES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

POLICIES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 5. Composite Backend
# ============================================================

backend = CompositeBackend(

    # --------------------------------------------------------
    # Anything that doesn't match a filesystem route
    # goes into agent state.
    # --------------------------------------------------------

    default=StateBackend(),

    routes={

        # ----------------------------------------------------
        # /memories/*
        #
        # Virtual:
        #     /memories/birthday.txt
        #
        # Physical:
        #     workspace/memories/birthday.txt
        # ----------------------------------------------------

        "/memories/": FilesystemBackend(
            root_dir=str(MEMORIES_DIR),
            virtual_mode=True,
        ),

        # ----------------------------------------------------
        # /policies/*
        #
        # Virtual:
        #     /policies/leave_policy.txt
        #
        # Physical:
        #     workspace/policies/leave_policy.txt
        # ----------------------------------------------------

        "/policies/": FilesystemBackend(
            root_dir=str(POLICIES_DIR),
            virtual_mode=True,
        ),

        # ----------------------------------------------------
        # General workspace
        #
        # This route lets the agent read other workspace files.
        # Permission rules below prevent arbitrary writes.
        # ----------------------------------------------------

        "/": FilesystemBackend(
            root_dir=str(WORKSPACE),
            virtual_mode=True,
        ),
    },
)


# ============================================================
# 6. Filesystem Permissions
# ============================================================

permissions = [FilesystemPermission(operations=["write","read"],paths=["/**"],mode="deny",),]


# ============================================================
# 7. Create Deep Agent
# ============================================================

agent = create_deep_agent(

    model=model,

    backend=backend,

    permissions=permissions,

    checkpointer=checkpointer,
)


# ============================================================
# 8. Thread Management
# ============================================================

def create_thread():
    return f"sajid-session-{uuid.uuid4().hex[:8]}"


THREAD_ID = create_thread()


def get_config():
    return {
        "configurable": {
            "thread_id": THREAD_ID,
        }
    }


# ============================================================
# 9. Agent Runner
# ============================================================

def ask_agent(message: str):

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ]
        },
        config=get_config(),
    )

    messages = result.get(
        "messages",
        [],
    )

    if not messages:
        print("\nNo response received.")
        return result

    last_message = messages[-1]

    print("\nAgent:")

    content = getattr(
        last_message,
        "content",
        last_message,
    )

    print(content)

    return result


# ============================================================
# 10. Physical Workspace Viewer
# ============================================================

def show_workspace():

    print("\nPhysical workspace:")
    print(
        f"{WORKSPACE}"
    )

    print()

    files = list(
        WORKSPACE.rglob("*")
    )

    if not files:
        print("(empty)")
        return

    for path in files:

        relative = path.relative_to(
            WORKSPACE
        )

        if path.is_dir():
            print(
                f"[DIR]  {relative}"
            )
        else:
            print(
                f"[FILE] {relative}"
            )


# ============================================================
# 11. CLI
# ============================================================

def main():

    global THREAD_ID

    print("=" * 70)

    print(
        " Deep Agent + CompositeBackend + FilesystemBackend "
    )

    print("=" * 70)

    print(
        f"""
Workspace:
{WORKSPACE}

Memories:
{MEMORIES_DIR}

Policies:
{POLICIES_DIR}

Thread:
{THREAD_ID}


Commands:

exit
    Quit

new
    Create a new conversation

thread
    Show current thread

workspace
    Show physical workspace
"""
    )

    while True:

        try:

            user_input = input(
                "\nYou: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print(
                "\nExiting..."
            )

            break


        if not user_input:
            continue


        # ====================================================
        # EXIT
        # ====================================================

        if user_input.lower() in {
            "exit",
            "quit",
        }:

            print(
                "Goodbye!"
            )

            break


        # ====================================================
        # THREAD
        # ====================================================

        if user_input.lower() == "thread":

            print(
                f"\nCurrent thread: {THREAD_ID}"
            )

            continue


        # ====================================================
        # NEW THREAD
        # ====================================================

        if user_input.lower() == "new":

            THREAD_ID = create_thread()

            print(
                f"\nNew thread created:"
                f" {THREAD_ID}"
            )

            continue


        # ====================================================
        # WORKSPACE
        # ====================================================

        if user_input.lower() == "workspace":

            show_workspace()

            continue


        # ====================================================
        # AGENT
        # ====================================================

        try:

            ask_agent(
                user_input
            )

        except Exception as e:

            print(
                "\nAgent Error:"
            )

            print(
                type(e).__name__
            )

            print(
                e
            )


# ============================================================
# 12. Start
# ============================================================

if __name__ == "__main__":
    main()