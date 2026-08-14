from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from rich import print
from dotenv import load_dotenv
import os
import base64
import httpx
load_dotenv()

model=ChatOpenAI(
    model="MiniMax-M3",
    base_url=os.getenv("MINIMAX_BASE_URL"),
    api_key=os.getenv("MINIMAX_API_KEY"),
)

agent=create_deep_agent(model=model)


def fetch_video_as_base64(url: str) -> str:
    """Download a remote video file and return its base64-encoded contents."""
    response = httpx.get(url)
    response.raise_for_status()
    return base64.b64encode(response.content).decode("utf-8")


print("[bold green]Invoking agent with multimodal input:: video from internet...[/bold green]")

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this video."},
            {
                "type": "video",
                "base64": fetch_video_as_base64("https://example.com/path/to/video.mp4"),
                "mime_type": "video/mp4",
            },
        ],
    }],
})


print(result)


# Read the local video file and encode it to base64
video_path1 = r"C:\Users\miyas\Downloads\sample1.mp4"
video_path2 = r"C:\Users\miyas\Downloads\sample2.mov"
with open(video_path1, "rb") as video_file:
    video_base64_1 = base64.b64encode(video_file.read()).decode("utf-8")
with open(video_path2, "rb") as video_file:
    video_base64_2 = base64.b64encode(video_file.read()).decode("utf-8")

print("[bold green]Invoking agent with multimodal input:: local video...[/bold green]")
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this video."},
            {
                "type": "video",
                "base64": video_base64_1,
                "mime_type": "video/mp4",
            },
            {
                "type": "video",
                "base64": video_base64_2,
                "mime_type": "video/quicktime",
            }
        ],
    }],
})

print(result["messages"][-1].content)


print("[bold green]Invoking agent with multimodal input:: video from hybrid...[/bold green]")

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this video."},
            {
                "type": "video",
                "base64": fetch_video_as_base64("https://example.com/path/to/video.mp4"),
                "mime_type": "video/mp4",
            },
             {
                "type": "video",
                "base64": video_base64_1,
                "mime_type": "video/mp4",
            }
        ],
    }],
})


print(result["messages"][-1].content)
