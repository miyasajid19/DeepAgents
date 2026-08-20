from rich import print
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from langchain_daytona import DaytonaSandbox
from daytona import Daytona
from pathlib import Path
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
load_dotenv()

client=Daytona()
sandbox = client.create()
backend=DaytonaSandbox(sandbox=sandbox)
print(f"Sandbox created: {sandbox.id}")

# files to upload
files=[
    Path(r"E:\Deep Agents\Projects\DataAnalyst\data.csv")
]

uploads=[]

for local_file in files:
    if not local_file.exists():
        print(f"Skipping: {local_file}")
        continue
    destination = (f"/home/daytona/{local_file.name}")
    uploads.append((destination,local_file.read_bytes(),))
    
    
print(uploads)

print("\nUploading files...")
backend.upload_files(uploads)




checkpointer = InMemorySaver()


model=ChatOpenAI(
    model_name="MiniMax-M3",
    api_key=os.environ.get("MINIMAX_API_KEY",None),
    base_url=os.environ.get("MINIMAX_BASE_URL",None),
)


agent=create_deep_agent(
    model=model,
    backend=backend,
    checkpointer=checkpointer,
    middleware=[TodoListMiddleware()],
)

thread_id=uuid7()

config={
    "configurable":{
        "thread_id":thread_id,
    }
}

input_message=HumanMessage(content="Analyze ./data.csv in the current dir, create codes/*.py  and perform EDA there .and save the result in plot/* and also generate a summary of the data in summary.txt")


stream=agent.stream_events({"messages":[input_message],},config=config,version='v3')

for snapshot in stream.values:
    print(snapshot["messages"][-1].content)

print("\nDownloading files...")
DOWNLOAD_DIR = Path("downloads/artifacts").resolve()
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Recursively collect everything the agent produced under codes/ and plot/,
# plus the top-level summary.txt. Using `find` over execute() avoids relying
# on backend.ls()'s structured shape — we just need raw paths.
REMOTE_ROOTS = ["/home/daytona/codes", "/home/daytona/plot"]
ALWAYS_PULL = ["/home/daytona/summary.txt"]


def find_remote_files(remote_dir: str) -> list[str]:
    """List regular files under `remote_dir` using `find`; empty if dir absent."""
    res = backend.execute(f"find {remote_dir} -type f 2>/dev/null")
    if res.exit_code != 0:
        return []
    return [line for line in res.output.splitlines() if line.strip()]


remote_files: list[str] = []
for root in REMOTE_ROOTS:
    remote_files.extend(find_remote_files(root))
for must_have in ALWAYS_PULL:
    if must_have not in remote_files:
        remote_files.append(must_have)

if not remote_files:
    print("[yellow]No files found under codes/, plot/, or summary.txt[/yellow]")
else:
    print(f"[blue]Pulling {len(remote_files)} file(s):[/blue]")
    for rf in remote_files:
        print(f"  • {rf}")

    # Single batched call — partial-success per file.
    responses = backend.download_files(remote_files)

    saved, failed = [], []
    for resp in responses:
        # Strip the sandbox root so subdir layout is preserved locally:
        #   /home/daytona/codes/eda.py  ->  DOWNLOAD_DIR/codes/eda.py
        rel = resp.path.removeprefix("/home/daytona/").lstrip("/")
        local_path = DOWNLOAD_DIR / rel
        if resp.error or resp.content is None:
            failed.append((resp.path, resp.error or "empty_content"))
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(resp.content)
        saved.append(local_path)

    print(f"\n[green]Saved {len(saved)} file(s) to {DOWNLOAD_DIR}[/green]")
    for p in saved:
        print(f"  ✓ {p}")
    if failed:
        print(f"\n[red]Failed {len(failed)} file(s):[/red]")
        for path, err in failed:
            print(f"  ✗ {path}  ({err})")
print("[red]Stopping and deleting sandbox...[/red]")
print("[red]Stopping sandbox...[/red]")

sandbox.stop()
print("[red]Stopped sandbox.[/red]")
print("[red]Deleting sandbox...[/red]")
client.delete(sandbox, wait=True)
print("[red]Deleted sandbox.[/red]")