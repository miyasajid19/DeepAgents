from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

# Load .env from the script directory explicitly
env_path = Path(__file__).parent / ".env"
loaded = load_dotenv(dotenv_path=env_path)

print(f".env loaded: {loaded} from {env_path}")

MINIMAX_API_KEY = (os.getenv("MINIMAX_API_KEY") or "").strip()
BASE_URL = (os.getenv("MINIMAX_BASE_URL") or "").strip()
MODEL = (os.getenv("MINIMAX_MODEL") or "").strip()

print(f"API key present: {bool(MINIMAX_API_KEY)}")
print(f"BASE_URL: {BASE_URL!r}")
print(f"MODEL: {MODEL!r}")

if not MINIMAX_API_KEY or not BASE_URL or not MODEL:
    raise ValueError("Missing env vars. Check .env keys/format/location.")

model = ChatOpenAI(
    model_name=MODEL,
    api_key=MINIMAX_API_KEY,
    base_url=BASE_URL,
    temperature=0.7
)




def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_deep_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
print(agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
))