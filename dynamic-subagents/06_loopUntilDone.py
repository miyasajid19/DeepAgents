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

# Virtual filesystem — the main agent stores the accumulating findings.json
# here across loop iterations so dedup state survives between analyzer calls.
backend = FilesystemBackend(root_dir="./dynamic-subagents", virtual_mode=True)

# Stateless analyzer: handles ONE batch of files per invocation.
# Returns findings as a JSON array so the main agent can dedupe deterministically.
analyzer = {
    "name": "analyzer",
    "description": "Scans one batch of files for dead code and returns findings as JSON",
    "system_prompt": (
        "You are a code analyst specializing in dead code detection. "
        "Given a batch of file paths, read each one and look for:\n"
        "  - unused exports (functions/classes/vars defined but never imported)\n"
        "  - unreachable functions (never called from anywhere)\n"
        "  - orphaned modules (no importers, no entry-point use)\n"
        "  - unreachable branches (code after a return/raise that can never run)\n\n"
        "For each finding return a JSON object with:\n"
        "  {file, line, symbol, category, evidence}\n\n"
        "Return your findings as a JSON array. If you find nothing in the "
        "requested batch, return [] — never fabricate findings."
    ),
}

agent = create_deep_agent(
    model=model,
    subagents=[analyzer],
    middleware=[CodeInterpreterMiddleware()],
    backend=backend,
)


if __name__ == "__main__":
    print("====" * 10)
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                "Run a loop-until-done dead-code sweep over './dynamic-subagents'.\n\n"
                "Process:\n"
                "1. Glob all .py files in the directory and partition them into "
                "   batches (3 files per batch).\n"
                "2. Initialize an empty `findings.json` in the virtual filesystem "
                "   with shape {findings: []}.\n"
                "3. For each batch: invoke the `analyzer` subagent, then use the "
                "   code interpreter to read `findings.json`, dedupe the new "
                "   findings against it (key = file+line+symbol), append only "
                "   fresh ones, and write it back. Report `new_count`.\n"
                "4. If a batch returns new_count == 0, mark it exhausted.\n"
                "5. Loop until every batch is exhausted.\n"
                "6. Output the final accumulated findings grouped by category "
                "   (unused exports, unreachable functions, orphaned modules, "
                "   unreachable branches)."
            )
        }]
    })
    print(result)
