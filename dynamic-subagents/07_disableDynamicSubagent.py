from rich import print
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware
load_dotenv()
model=ChatOpenAI(
    model_name="Minimax-M3",
    api_key=os.environ.get("MINIMAX_API_KEY"),
    base_url=os.environ.get("MINIMAX_BASE_URL"),
)
subagent_model=ChatOpenAI(
    model_name="Minimax-M2.7",
    api_key=os.environ.get("MINIMAX_API_KEY"),
    base_url=os.environ.get("MINIMAX_BASE_URL"),
)
joke_subagent = {
    "name": "joke_subagent",
    "description": "A subagent that tells jokes but joke is always lame and not funny",
    "system_prompt": "You are a comedy specialist that exclusively tells lame, unfunny jokes.",
    "model": subagent_model,
}

coder_agent = {
    "name": "coder_agent",
    "description": "A subagent that helps with coding tasks",
    "system_prompt": "You are an expert programming assistant that writes clean, concise code.",
    "model": subagent_model,
}

agent=create_deep_agent(
    model=model,
    subagents=[joke_subagent,coder_agent],
    middleware=[CodeInterpreterMiddleware(subagents=False)],
)


if __name__ == "__main__":
    print(agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Run a workflow to write a python function to add two numbers using the coder_agent, and then evaluate or joke about it using the joke_subagent."
        }]
    }))
    print("===="*10)
    print(agent.invoke({"messages":[{"role":"user","content":"write a python function to add two numbers"}]}))
    
    
    
    
    
    # To trigger dynamic subagents, prompt the agent with the word “workflow”