"""Remote researcher graph hosted by the LangGraph dev server.

Served as the `researcher` assistant (graph_id). The async subagent middleware
calls `client.runs.create(assistant_id="researcher", ...)` against this server.
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
    temperature=0.7,
)


graph = create_agent(
    model=_model,
    tools=[],
    system_prompt=(
        "You are a research specialist. When the user asks for information on a "
        "topic, return a concise, well-structured summary (3–6 bullet points). "
        "Cite concrete trends, names, and dates when possible. Do not use tools."
    ),
)
