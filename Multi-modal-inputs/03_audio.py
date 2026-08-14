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


def fetch_audio_as_base64(url: str) -> str:
    """Download a remote audio file and return its base64-encoded contents."""
    response = httpx.get(url)
    response.raise_for_status()
    return base64.b64encode(response.content).decode("utf-8")


print("[bold green]Invoking agent with multimodal input:: audio from internet...[/bold green]")

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this audio."},
            {
                "type": "audio",
                "base64": fetch_audio_as_base64("https://example.com/path/to/audio.wav"),
                "mime_type": "audio/wav",
            },
        ],
    }],
})


print(result)


# Read the local audio file and encode it to base64
audio_path1 = r"C:\Users\miyas\Downloads\sample1.wav"
audio_path2 = r"C:\Users\miyas\Downloads\sample2.mp3"
with open(audio_path1, "rb") as audio_file:
    audio_base64_1 = base64.b64encode(audio_file.read()).decode("utf-8")
with open(audio_path2, "rb") as audio_file:
    audio_base64_2 = base64.b64encode(audio_file.read()).decode("utf-8")

print("[bold green]Invoking agent with multimodal input:: local audio...[/bold green]")
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this audio."},
            {
                "type": "audio",
                "base64": audio_base64_1,
                "mime_type": "audio/wav",
            },
            {
                "type": "audio",
                "base64": audio_base64_2,
                "mime_type": "audio/mpeg",
            }
        ],
    }],
})

print(result["messages"][-1].content)


print("[bold green]Invoking agent with multimodal input:: audio from hybrid...[/bold green]")

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this audio."},
            {
                "type": "audio",
                "base64": fetch_audio_as_base64("https://example.com/path/to/audio.wav"),
                "mime_type": "audio/wav",
            },
             {
                "type": "audio",
                "base64": audio_base64_1,
                "mime_type": "audio/wav",
            }
        ],
    }],
})


print(result["messages"][-1].content)
