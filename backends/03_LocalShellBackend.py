import os
import uuid

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import InMemorySaver

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend


# ============================================================
# 1. Environment
# ============================================================

load_dotenv()


MINIMAX_API_KEY = os.getenv(
    "MINIMAX_API_KEY"
)

MINIMAX_BASE_URL = os.getenv(
    "MINIMAX_BASE_URL"
)


if not MINIMAX_API_KEY:
    raise ValueError(
        "MINIMAX_API_KEY is not set"
    )


if not MINIMAX_BASE_URL:
    raise ValueError(
        "MINIMAX_BASE_URL is not set"
    )



# ============================================================
# 2. LLM Model
# ============================================================


model = ChatOpenAI(
    model="MiniMax-M3",
    base_url=MINIMAX_BASE_URL,
    api_key=MINIMAX_API_KEY,
    temperature=0.2,
)



# ============================================================
# 3. LangGraph Checkpointer
# ============================================================


"""
Temporary memory.

Stored:
    - messages
    - agent state
    - tool calls

Lost after program restart.
"""

checkpointer = InMemorySaver()



# ============================================================
# 4. Agent Workspace Backend
# ============================================================


os.makedirs(
    "workspace",
    exist_ok=True
)


backend = LocalShellBackend(
    root_dir=".",
    virtual_mode=True,
    env={
        "PATH": "/usr/bin:/bin"
    }
)



# ============================================================
# 5. Create Deep Agent
# ============================================================


agent = create_deep_agent(
    model=model,
    backend=backend,
    checkpointer=checkpointer,
)



# ============================================================
# 6. Thread Management
# ============================================================


def create_thread():

    return (
        f"sajid-session-"
        f"{uuid.uuid4().hex[:8]}"
    )


THREAD_ID = create_thread()



def get_config():

    return {
        "configurable": {
            "thread_id": THREAD_ID
        }
    }



# ============================================================
# 7. Agent Runner
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
        config=get_config()
    )


    messages = result.get(
        "messages",
        []
    )


    if not messages:

        print(
            "No response received"
        )

        return



    last_message = messages[-1]


    print(
        "\nAgent:"
    )


    if hasattr(
        last_message,
        "content"
    ):

        print(
            last_message.content
        )

    else:

        print(
            last_message
        )



    return result



# ============================================================
# 8. CLI
# ============================================================


def main():


    global THREAD_ID


    print(
        "=" * 70
    )

    print(
        " DeepAgents + MiniMax + LocalShellBackend "
    )

    print(
        "=" * 70
    )


    print(
        f"""
Thread:
{THREAD_ID}


Commands:

exit
    Quit

new
    Create new conversation

thread
    Show current thread

"""
    )


    while True:


        try:

            user_input = input(
                "\nYou: "
            ).strip()


        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                "\nExiting..."
            )

            break



        if not user_input:

            continue



        # -----------------------------
        # Exit
        # -----------------------------

        if user_input.lower() in {
            "exit",
            "quit"
        }:

            print(
                "Goodbye!"
            )

            break



        # -----------------------------
        # Current thread
        # -----------------------------


        if user_input.lower() == "thread":

            print(
                f"\nCurrent thread:"
                f" {THREAD_ID}"
            )

            continue



        # -----------------------------
        # New conversation
        # -----------------------------


        if user_input.lower() == "new":


            THREAD_ID = create_thread()


            print(
                "\nNew thread:"
            )

            print(
                THREAD_ID
            )


            continue



        # -----------------------------
        # Agent execution
        # -----------------------------


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
# Start
# ============================================================


if __name__ == "__main__":

    main()