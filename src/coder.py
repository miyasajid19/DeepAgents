"""Remote coder graph hosted by the LangGraph dev server.

Served as the `coder` assistant (graph_id). The async subagent middleware
calls `client.runs.create(assistant_id="coder", ...)` against this server.
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
    temperature=0.4,
)


graph = create_agent(
    model=_model,
    tools=[],
    system_prompt=(
        "You are a coding specialist. When asked for a code snippet, return a "
        "single fenced code block (```python ... ```) with the requested code. "
        "Keep it minimal and runnable. Do not use tools."
    ),
)
