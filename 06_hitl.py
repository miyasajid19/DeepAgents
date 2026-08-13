from langchain.tools import tool
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from rich import print
from dotenv import load_dotenv
import os
load_dotenv()

@tool
def remove_file(path: str) -> str:
    """Delete a file from the filesystem."""
    return f"Deleted {path}"


@tool
def fetch_file(path: str) -> str:
    """Read a file from the filesystem."""
    return f"Contents of {path}"


@tool
def notify_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Sent email to {to}"


# Checkpointer is REQUIRED for human-in-the-loop
checkpointer = MemorySaver()
model=ChatOpenAI(
    base_url=os.getenv("MINIMAX_BASE_URL"),
    api_key=os.getenv("MINIMAX_API_KEY"),
    model="MiniMax-M3",
)
agent = create_deep_agent(
    model=model,
    tools=[remove_file, fetch_file, notify_email],
    interrupt_on={
        "remove_file": {"allowed_decisions": ["approve", "reject"]},
        "fetch_file": False,  # No interrupts needed
        "notify_email": {"allowed_decisions": ["approve", "reject"]},
    },
    checkpointer=checkpointer,
)


def _decision_prompt(action_name: str, args: dict, allowed_decisions: list[str]) -> dict:
    """Prompt user to approve or reject a pending tool call."""
    allowed = [d for d in allowed_decisions if d in {"approve", "reject"}]
    if not allowed:
        allowed = ["approve", "reject"]

    print("\n[bold yellow]Approval Required[/bold yellow]")
    print(f"Tool: [cyan]{action_name}[/cyan]")
    print(f"Args: {args}")
    print(f"Allowed: {', '.join(allowed)}")

    while True:
        choice = input("Decision (approve/reject): ").strip().lower()
        if choice not in allowed:
            print("Invalid choice. Type 'approve' or 'reject'.")
            continue

        if choice == "approve":
            return {"type": "approve"}

        reason = input("Optional rejection reason (press Enter to skip): ").strip()
        decision = {"type": "reject"}
        if reason:
            decision["message"] = reason
        return decision


def _resume_with_decisions(result: dict, config: dict) -> dict:
    """Resolve pending interrupts by collecting human decisions and resuming."""
    while result.get("__interrupt__"):
        interrupts = result["__interrupt__"]

        if len(interrupts) == 1:
            interrupt_obj = interrupts[0]
            payload = getattr(interrupt_obj, "value", {})
            action_requests = payload.get("action_requests", [])
            review_configs = payload.get("review_configs", [])

            decisions = []
            for action, review in zip(action_requests, review_configs, strict=False):
                decisions.append(
                    _decision_prompt(
                        action_name=action.get("name", "unknown"),
                        args=action.get("args", {}),
                        allowed_decisions=review.get("allowed_decisions", []),
                    )
                )

            result = agent.invoke(Command(resume={"decisions": decisions}), config=config)
            continue

        resume_map = {}
        for interrupt_obj in interrupts:
            interrupt_id = getattr(interrupt_obj, "id", None)
            payload = getattr(interrupt_obj, "value", {})
            action_requests = payload.get("action_requests", [])
            review_configs = payload.get("review_configs", [])

            decisions = []
            for action, review in zip(action_requests, review_configs, strict=False):
                decisions.append(
                    _decision_prompt(
                        action_name=action.get("name", "unknown"),
                        args=action.get("args", {}),
                        allowed_decisions=review.get("allowed_decisions", []),
                    )
                )

            if interrupt_id:
                resume_map[interrupt_id] = {"decisions": decisions}

        result = agent.invoke(Command(resume=resume_map), config=config)

    return result

if __name__ == "__main__":
    run_config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Please remove the file /tmp/test.txt, fetch the file /tmp/data.txt, and send an email to user@example.com with subject 'Run Update' and body 'All tasks completed.'"
                }
            ]
        },
        config=run_config,
    )

    result = _resume_with_decisions(result, run_config)

    print("\n[bold green]Final Result[/bold green]")
    print(result)
    