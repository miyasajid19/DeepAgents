from rich import print
from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()
agent = create_deep_agent(
    model=ChatOpenAI(
        model_name="Minimax-M3",
        api_key=os.environ.get("MINIMAX_API_KEY"),
        base_url=os.environ.get("MINIMAX_BASE_URL"),
    ),
    subagents=[
        {
            "name": "bug-fixer",
            "description": "Investigates bug reports and provides reproduction steps",
            "system_prompt": "You are a bug triage specialist. Investigate each bug report and provide clear reproduction steps.",
        },
        {
            "name": "feature-analyst",
            "description": "Evaluates feature requests for feasibility and effort",
            "system_prompt": "You are a product analyst. Evaluate each feature request for technical feasibility, estimated effort, and potential impact.",
        },
        {
            "name": "support-agent",
            "description": "Answers user questions based on documentation",
            "system_prompt": "You are a support specialist. Answer user questions clearly based on the available documentation.",
        },
    ],
    middleware=[CodeInterpreterMiddleware()],
)

if __name__ == "__main__":
    print("===="*10)
    print(agent.invoke({"messages":[{"role":"user","content":"Investigate the bug report for the login issue."}]}))
    print("===="*10)
    print(agent.invoke({"messages":[{"role":"user","content":"Evaluate the feature request for dark mode."}]}))
    print("===="*10)
    print(agent.invoke({"messages":[{"role":"user","content":"Answer the user question about password reset."}]}))