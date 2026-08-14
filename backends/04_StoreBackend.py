import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from deepagents import create_deep_agent
from deepagents.backends import StoreBackend


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

backend = StoreBackend(
    namespace=lambda rt: (
        rt.server_info.user.identity 
        if rt.server_info and rt.server_info.user 
        else "default-user",
    ),
)




# ------------------------------ 
# common namespaces format 
# ------------------------------ 

# from deepagents.backends import StoreBackend

# # Per-user: each user gets their own isolated storage
# backend = StoreBackend(
#     namespace=lambda rt: (rt.server_info.user.identity,),
# )

# # Per-assistant: all users of the same assistant share storage
# backend = StoreBackend(
#     namespace=lambda rt: (
#         rt.server_info.assistant_id,
#     ),
# )

# # Per-thread: storage scoped to a single conversation
# backend = StoreBackend(
#     namespace=lambda rt: (
#         rt.execution_info.thread_id,
#     ),
# )




# ============================================================
# 5. Create Deep Agent
# ============================================================

agent = create_deep_agent(
    model=model,
    backend=backend,
    checkpointer=checkpointer,
    store=InMemoryStore(),   # good for local development but not for production. Langsmith auto adds it
)

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
print("Deep Agents + StoreBackend + InMemorySaver")
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