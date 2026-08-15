from dotenv import load_dotenv
import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware


# ============================================================
# 1. Environment
# ============================================================

env_path = Path(__file__).parent / ".env"

loaded = load_dotenv(dotenv_path=env_path)

print(f".env loaded: {loaded} from {env_path}")


MINIMAX_API_KEY = (os.getenv("MINIMAX_API_KEY") or "").strip()

BASE_URL = (os.getenv("MINIMAX_BASE_URL") or "").strip()

MODEL = (os.getenv("MINIMAX_MODEL") or "").strip()


print(f"API key present: {bool(MINIMAX_API_KEY)}")

print(f"BASE_URL: {BASE_URL!r}")

print(f"MODEL: {MODEL!r}")


if not MINIMAX_API_KEY:
    raise ValueError("MINIMAX_API_KEY is missing")

if not BASE_URL:
    raise ValueError("MINIMAX_BASE_URL is missing")

if not MODEL:
    raise ValueError("MINIMAX_MODEL is missing")


# ============================================================
# 2. LLM
# ============================================================

model = ChatOpenAI(
    model=MODEL,
    api_key=MINIMAX_API_KEY,
    base_url=BASE_URL,
    temperature=0.2,
)


# ============================================================
# 3. Deep Agent + Code Interpreter
# ============================================================

agent = create_deep_agent(
    model=model,

    middleware=[
        CodeInterpreterMiddleware()
    ],

    system_prompt="""
You are a helpful AI assistant.

You have access to a code interpreter.

Use the code interpreter when the user asks you
to perform calculations, execute code, analyze data,
or verify a computational result.

Actually execute code when appropriate.
Do not claim that code was executed unless it
was actually executed.
""",
)


# ============================================================
# 4. Test
# ============================================================

print("\n" + "=" * 70)

print(    "DeepAgents + Code Interpreter")

print("=" * 70)


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": """
Use the code interpreter to calculate:

1. The sum of numbers from 1 to 1000.
2. The average.
3. The standard deviation.

Actually execute the code and give me the results.
""",
            }
        ]
    }
)


# ============================================================
# 5. Response
# ============================================================

print("\nAgent:")

print(result["messages"][-1].content)