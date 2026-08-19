"""Supervisor graph hosted by the LangGraph dev server.

This is the entry-point graph that the dev server exposes. It is usually not
invoked by `01_quickUsage.py` (the main agent runs locally with `create_deep_agent`
and dispatches to the other two hosted graphs via `AsyncSubAgent`). It is kept
here so `langgraph.json` resolves and the server boots cleanly, and so the
supervisor graph can be tested independently via the dev UI.
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

_model = ChatOpenAI(
    model=os.environ.get("MINIMAX_MODEL", "MiniMax-M3"),
    api_key=os.environ.get("MINIMAX_API_KEY"),
    base_url=os.environ.get("MINIMAX_BASE_URL"),
    temperature=0.5,
)


graph = create_agent(
    model=_model,
    tools=[],
    system_prompt=(
        "You are a friendly supervisor. Briefly acknowledge the user's request "
        "and explain that the actual work is being delegated to specialised "
        "remote subagents (researcher and coder)."
    ),
)
