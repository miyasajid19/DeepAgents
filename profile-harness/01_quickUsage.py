from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.messages import HumanMessage
import os
from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)
# ============================================================
# Environment
# ============================================================
load_dotenv()
# ============================================================
# Model
# ============================================================
model = ChatOpenAI(
    model="MiniMax-M3",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
)
# ============================================================
# Register Harness Profile
register_harness_profile(
    "openai:MiniMax-M3",
    HarnessProfile(
        system_prompt_suffix="""Respond in nepali roman.""",

        excluded_tools={"execute"},

        excluded_middleware={"SummarizationMiddleware"},

        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    ),
)
# ============================================================
# Create Agent
# ============================================================
agent = create_deep_agent(model=model,)
# ============================================================
# Run
# ============================================================

result = agent.invoke(
    {
        "messages":[
            HumanMessage(
                content=
                "Write an essay on the topic of artificial intelligence."
            )
        ],
    }
)
print(result)