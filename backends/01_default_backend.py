import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from deepagents import create_deep_agent
from deepagents.backends import StateBackend


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
)


# ============================================================
# 3. In-memory checkpointer
# ============================================================

checkpointer = InMemorySaver()


# ============================================================
# 4. State-backed filesystem
# ============================================================

backend = StateBackend()


# ============================================================
# 5. Create Deep Agent
# ============================================================

agent = create_deep_agent(
    model=model,
    backend=backend,
    checkpointer=checkpointer,
)
agent1 = create_deep_agent(
    model=model,
    checkpointer=checkpointer,
)

print("agent and agent1 are created with state backend and in-memory checkpointer.StateBackend is the default even though you don't pass it in the create_deep_agent function. You can use the agent to make requests and it will automatically save the state to the backend and checkpointer.")

# ============================================================
# 6. Thread configuration
# ============================================================

THREAD_ID = "sajid-session-1"

config = {
    "configurable": {
        "thread_id": THREAD_ID,
    }
}


# ============================================================
# 7. Helper
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
        config=config,
    )

    messages = result.get("messages", [])

    if messages:
        last_message = messages[-1]

        print("\nAgent:")

        content = last_message.content

        if isinstance(content, str):
            print(content)
        else:
            print(content)

    return result


# ============================================================
# 8. REPL
# ============================================================

print("=" * 65)
print("Deep Agents + StateBackend + InMemorySaver")
print("=" * 65)

print(f"Thread: {THREAD_ID}")
print()
print("Commands:")
print("  exit       -> quit")
print("  new        -> create a new conversation")
print("  thread     -> show current thread")
print("=" * 65)


while True:

    try:
        user_input = input("\nYou: ").strip()

    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        break

    if not user_input:
        continue

    # --------------------------------------------------------
    # Exit
    # --------------------------------------------------------

    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break


    # --------------------------------------------------------
    # Show current thread
    # --------------------------------------------------------

    if user_input.lower() == "thread":
        print(f"\nCurrent thread: {THREAD_ID}")
        continue


    # --------------------------------------------------------
    # Create new thread
    # --------------------------------------------------------

    if user_input.lower() == "new":

        import uuid

        THREAD_ID = f"sajid-session-{uuid.uuid4().hex[:8]}"

        config = {
            "configurable": {
                "thread_id": THREAD_ID,
            }
        }

        print(f"\nNew thread created: {THREAD_ID}")

        continue


    # --------------------------------------------------------
    # Normal agent invocation
    # --------------------------------------------------------

    try:

        ask_agent(user_input)

    except Exception as e:

        print("\nAgent Error:")
        print(type(e).__name__)
        print(e)