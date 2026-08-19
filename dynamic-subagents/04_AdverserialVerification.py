from rich import print
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware
from deepagents.backends import FilesystemBackend

load_dotenv()

model = ChatOpenAI(
    model_name="Minimax-M3",
    api_key=os.environ.get("MINIMAX_API_KEY"),
    base_url=os.environ.get("MINIMAX_BASE_URL"),
)

# Virtual filesystem scoped to ./dynamic-subagents — agents read/write
# inside the sandbox without touching the real host disk.
backend = FilesystemBackend(root_dir="./dynamic-subagents", virtual_mode=True)

reviewer = {
    "name": "reviewer",
    "description": "First-pass security scan — broad recall, may include false positives",
    "system_prompt": (
        "You are a security auditor performing a broad first-pass scan. "
        "Glob the target file, read it carefully, and report EVERY potential "
        "vulnerability you find — even uncertain ones. False positives are "
        "acceptable here; missed issues are not. For each finding, cite "
        "file, line range, and a one-sentence description. Prefer to over-report."
    ),
}

verifier = {
    "name": "verifier",
    "description": "Independently verifies whether a reported vulnerability is real and exploitable",
    "system_prompt": (
        "You are a skeptical security verification specialist. "
        "For each finding produced by the reviewer, independently re-inspect "
        "the cited lines and decide whether the issue is REAL or FALSE POSITIVE. "
        "Demand evidence: can the code path actually be reached? Is the input "
        "controlled by an attacker? Is the impact realistic? "
        "Be adversarial — assume the reviewer is wrong until you can confirm otherwise. "
        "When uncertain, run a small reproducer to settle it. "
        "Output only verified issues with a confidence level (high/medium/low)."
    ),
}

agent = create_deep_agent(
    model=model,
    subagents=[reviewer, verifier],
    middleware=[CodeInterpreterMiddleware()],
    backend=backend,
)


if __name__ == "__main__":
    print("====" * 10)
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                "Adversarially verify 'dynamic-subagents/02_classifyAndAct.py'. "
                "Have the reviewer do a broad first-pass scan and list every "
                "potential vulnerability. Then have the verifier independently "
                "re-check each finding and report only the confirmed, "
                "exploitable issues with confidence levels."
            )
        }]
    })
    print(result)