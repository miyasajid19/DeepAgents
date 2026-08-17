from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.messages import HumanMessage
import yaml
from deepagents import HarnessProfileConfig, register_harness_profile
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
# Register Harness Profile



with open("profile-harness/openai.yaml") as f:
    register_harness_profile(
        "openai:MiniMax-M3",
        HarnessProfileConfig.from_dict(yaml.safe_load(f))
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