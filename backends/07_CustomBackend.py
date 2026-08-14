# this is a method but might not be correct

from pathlib import Path

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import EditResult, GlobResult, GrepResult, LsResult, ReadResult, WriteResult


class DeleteAwareFilesystemBackend(FilesystemBackend):
    """Filesystem backend with a real delete operation for agent tools."""

    def __init__(self, root_dir: str = "."):
        super().__init__(root_dir=root_dir, virtual_mode=True)
        self.root_dir = Path(root_dir).resolve()

    def delete(self, file_path: str) -> dict:
        """Delete a file safely within the backend root directory."""
        clean_path = file_path.strip()
        if not clean_path or clean_path == "/":
            return {"error": "invalid_path", "deleted": False}

        target = (self.root_dir / clean_path.lstrip("/")).resolve()

        try:
            target.relative_to(self.root_dir)
        except ValueError:
            return {"error": "invalid_path", "deleted": False}

        if not target.exists():
            return {"error": "file_not_found", "deleted": False}

        if target.is_dir():
            return {"error": "is_directory", "deleted": False}

        target.unlink()
        return {"deleted": True, "path": clean_path}





import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend


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
backend = DeleteAwareFilesystemBackend(root_dir=".")



# ============================================================
# 5. Create Deep Agent
# ============================================================

agent = create_deep_agent(
    model=model,
    backend=backend,
    checkpointer=checkpointer,
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
print("Deep Agents + FilesystemBackend + InMemorySaver")
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