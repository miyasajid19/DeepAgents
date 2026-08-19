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

# Sandbox: only `glob` is permitted through the code interpreter.
# Reviewers can locate files but cannot execute arbitrary code or write back.
reviewer_sandbox = CodeInterpreterMiddleware(ptc=["glob"])

auth_reviewer = {
    "name": "auth-reviewer",
    "description": "Reviews code for authentication and authorization issues",
    "system_prompt": (
        "You are an authentication and authorization specialist. "
        "Glob to locate the target file, read it carefully, and report any "
        "issues with: missing auth checks, broken access control, "
        "privilege escalation, session handling, token validation, or IDOR. "
        "Cite line numbers and severity (critical/high/medium/low) per finding."
    ),
}

input_reviewer = {
    "name": "input-reviewer",
    "description": "Reviews code for input validation issues",
    "system_prompt": (
        "You are an input validation specialist. "
        "Glob to locate the target file, read it carefully, and report any "
        "issues with: SQL injection, command injection, path traversal, "
        "XSS, SSRF, deserialization, or unsafe regex. "
        "Cite line numbers and severity (critical/high/medium/low) per finding."
    ),
}

crypto_reviewer = {
    "name": "crypto-reviewer",
    "description": "Reviews code for cryptography and TLS",
    "system_prompt": (
        "You are a cryptography specialist. "
        "Glob to locate the target file, read it carefully, and report any "
        "issues with: weak algorithms, improper randomness, broken TLS, "
        "unsafe hashing, or missing encryption at rest or in transit. "
        "Cite line numbers and severity (critical/high/medium/low) per finding."
    ),
}

secrets_reviewer = {
    "name": "secrets-reviewer",
    "description": "Reviews code for hardcoded secrets and credential leakage",
    "system_prompt": (
        "You are a secrets management specialist. "
        "Glob to locate the target file, read it carefully, and report any: "
        "hardcoded API keys, tokens, passwords, private keys, committed .env "
        "values, or credentials logged in plaintext. "
        "Cite line numbers and severity (critical/high/medium/low) per finding."
    ),
}

# Synthesizer runs last — it only reads the structured outputs the
# reviewers already produced, so the sandbox restriction is fine here.
synthesizer = {
    "name": "synthesizer",
    "description": "Combines all reviewer findings into a single prioritized report",
    "system_prompt": (
        "You are a security report synthesizer. "
        "Read the findings produced by auth-reviewer, input-reviewer, "
        "crypto-reviewer, and secrets-reviewer. Deduplicate overlapping issues, "
        "resolve conflicts, and produce a single report grouped by severity "
        "(critical first, then high, medium, low). "
        "End with a short executive summary suitable for a PR description."
    ),
}

agent = create_deep_agent(
    model=model,
    subagents=[
        auth_reviewer,
        input_reviewer,
        crypto_reviewer,
        secrets_reviewer,
        synthesizer,
    ],
    middleware=[reviewer_sandbox],
    backend=FilesystemBackend(root_dir="./dynamic-subagents",virtual_mode=True),
)


if __name__ == "__main__":
    print("====" * 10)
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                "Run a fan-out security review of 'dynamic-subagents/02_classifyAndAct.py'. "
                "Dispatch auth-reviewer, input-reviewer, crypto-reviewer, and secrets-reviewer "
                "in parallel to inspect the file, then have the synthesizer combine their "
                "findings into one prioritized report."
            )
        }]
    })
    print(result)
