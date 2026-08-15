import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_quickjs import CodeInterpreterMiddleware
from deepagents import create_deep_agent
from rich import print


# ============================================================
# 1. Environment
# ============================================================

load_dotenv()

MINIMAX_API_KEY = (os.getenv("MINIMAX_API_KEY") or "").strip()

MINIMAX_BASE_URL = (os.getenv("MINIMAX_BASE_URL") or "").strip()

MINIMAX_MODEL = (os.getenv("MINIMAX_MODEL") or "").strip()


if not MINIMAX_API_KEY:
    raise ValueError("MINIMAX_API_KEY is missing")

if not MINIMAX_BASE_URL:
    raise ValueError("MINIMAX_BASE_URL is missing")

if not MINIMAX_MODEL:
    raise ValueError("MINIMAX_MODEL is missing")


# ============================================================
# 2. Model
# ============================================================

model = ChatOpenAI(
    model=MINIMAX_MODEL,
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL,
    temperature=0.2,
)


# ============================================================
# 3. Tool
# ============================================================

@tool
def web_search(query: str) -> str:
    """
    Search the web for information about a topic.

    This is a demonstration tool for PTC.
    In a production application this could call
    Tavily, Serper, Google, Bing, etc.
    """

    print(f"[cyan]web_search called:[/cyan] {query}")

    # --------------------------------------------------------
    # Demo response
    # --------------------------------------------------------

    return (
        f"Search results for '{query}':\n"
        f"- Result 1: Information about {query}\n"
        f"- Result 2: Best practices related to {query}\n"
        f"- Result 3: Recent developments related to {query}"
    )


# ============================================================
# 4. Code Interpreter + PTC
# ============================================================

middleware = CodeInterpreterMiddleware(ptc=["web_search"])


# ============================================================
# 5. Deep Agent
# ============================================================

agent = create_deep_agent(
    model=model,
    tools=[web_search],middleware=[middleware],

    system_prompt="""
You are a helpful research assistant.

You have access to a web_search tool.

The web_search tool is available through
Programmatic Tool Calling (PTC).

When multiple independent searches are required,
use the code interpreter to call web_search
programmatically and execute the searches
in parallel when possible.

For example:

const topics = [
    "retrieval",
    "memory",
    "evaluation"
];

const results = await Promise.all(
    topics.map((topic) =>
        tools.webSearch({
            query: `${topic} best practices`
        })
    )
);

results.join("\\n\\n");

Then analyze the combined results and provide
a concise answer.
"""
)


# ============================================================
# 6. Run Agent
# ============================================================

print()
print("[bold green]""============================================================""[/bold green]")

print("[bold green]""DeepAgents + PTC""[/bold green]")

print("[bold green]""============================================================""[/bold green]")

print()


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": """
Research these three topics:

1. Retrieval
2. Memory
3. LLM Evaluation

Search for best practices for each topic.

There are three independent searches,
so use PTC to execute them programmatically
and in parallel if possible.

Then combine the results and give me
a concise comparison.
"""
            }
        ]
    }
)


# ============================================================
# 7. Output
# ============================================================

print()

print("[bold blue]Agent:[/bold blue]")

print(result["messages"][-1].content)