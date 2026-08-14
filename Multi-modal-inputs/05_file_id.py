from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from rich import print
from dotenv import load_dotenv
import os
load_dotenv()

model=ChatOpenAI(
    model="MiniMax-M3",
    base_url=os.getenv("MINIMAX_BASE_URL"),
    api_key=os.getenv("MINIMAX_API_KEY"),
)

agent=create_deep_agent(model=model)


# Note: The `file_id` references assume the files have already been uploaded
# to the provider's file storage and returned an ID. Replace the IDs below
# with real ones from your provider before running.

print("[bold green]Invoking agent with multimodal input:: image via provider-managed file_id...[/bold green]")

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this image."},
            {"type": "image", "file_id": "file-abc123"},
        ],
    }],
})


print(result)


print("[bold green]Invoking agent with multimodal input:: pdf via provider-managed file_id...[/bold green]")

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this document."},
            {"type": "file", "file_id": "file-pdf-abc123"},
        ],
    }],
})


print(result["messages"][-1].content)


print("[bold green]Invoking agent with multimodal input:: audio via provider-managed file_id...[/bold green]")

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this audio."},
            {"type": "audio", "file_id": "file-audio-abc123"},
        ],
    }],
})


print(result["messages"][-1].content)


print("[bold green]Invoking agent with multimodal input:: video via provider-managed file_id...[/bold green]")

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this video."},
            {"type": "video", "file_id": "file-video-abc123"},
        ],
    }],
})


print(result["messages"][-1].content)


print("[bold green]Invoking agent with multimodal input:: mixed file_id across modalities...[/bold green]")

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare the content across these media files."},
            {"type": "image", "file_id": "file-image-abc123"},
            {"type": "file", "file_id": "file-pdf-abc123"},
            {"type": "audio", "file_id": "file-audio-abc123"},
            {"type": "video", "file_id": "file-video-abc123"},
        ],
    }],
})


print(result["messages"][-1].content)
