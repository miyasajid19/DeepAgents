from langchain.rate_limiters import InMemoryRateLimiter
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from deepagents import create_deep_agent
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # 1 request every 10s
    check_every_n_seconds=0.1,  # Check every 100ms whether allowed to make a request
    max_bucket_size=10,  # Controls the maximum burst size
)
load_dotenv()

model = ChatOpenAI(
    model="MiniMax-M3",
    base_url=os.getenv("MINIMAX_BASE_URL"),
    api_key=os.getenv("MINIMAX_API_KEY"),
    rate_limiter=rate_limiter,
)

agent = create_deep_agent(model=model, tools=[], interrupt_on={})
print("Agent created with rate limiting. You can now use the agent to make requests while respecting the rate limits.")
print(agent)
print("Rate limiter configuration:")
print(rate_limiter)
print(model)

for i in range(15):
    print(f"Making request {i+1}...")
    response = model.invoke("Hello, how are you?")
    print(f"Response: {response}")
    print("\n"*5)