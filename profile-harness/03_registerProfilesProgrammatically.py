from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.messages import HumanMessage
from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    ProviderProfile,
    register_harness_profile,
    register_provider_profile,
)
import os
from rich import print
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
# Register Harness & Provider Profiles
# ============================================================
PROFILE_KEY = "openai:MiniMax-M3"

register_harness_profile(
    PROFILE_KEY,
    HarnessProfile(
        system_prompt_suffix="""Batch independent tool calls in parallel. Respond in nepali roman.""",

        excluded_tools={"execute"},

        excluded_middleware={"SummarizationMiddleware"},

        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    ),
)

register_provider_profile(
    PROFILE_KEY,
    ProviderProfile(
        init_kwargs={"temperature": 0}
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
                "Write a short essay on the topic of artificial intelligence."
            )
        ],
    }
)
print(result)