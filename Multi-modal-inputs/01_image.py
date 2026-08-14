from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from rich import print
from dotenv import load_dotenv
import os
import base64
load_dotenv()

model=ChatOpenAI(
    model="MiniMax-M3",
    base_url=os.getenv("MINIMAX_BASE_URL"),
    api_key=os.getenv("MINIMAX_API_KEY"),
)

agent=create_deep_agent(model=model)

print("[bold green]Invoking agent with multimodal input:: image from internet...[/bold green]")

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this screenshot?"},
            {"type": "image", "url": r"https://m.media-amazon.com/images/M/MV5BZTNjOWI0ZTAtOGY1OS00ZGU0LWEyOWYtMjhkYjdlYmVjMDk2XkEyXkFqcGc@._V1_QL75_UY281_CR1,0,190,281_.jpg"},
        ],
    }],
})


print(result)


# Read the local image and encode it to base64
image_path1 = r"C:\Users\miyas\Downloads\1770815533408-certificate.png"
image_path2 = r"C:\Users\miyas\Downloads\NexHire.png"
with open(image_path1, "rb") as image_file:
    image_base64_1 = base64.b64encode(image_file.read()).decode("utf-8")
with open(image_path2, "rb") as image_file:
    image_base64_2 = base64.b64encode(image_file.read()).decode("utf-8")

print("[bold green]Invoking agent with multimodal input:: local image...[/bold green]")
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this screenshot?"},
            {
                "type": "image",
                "base64": image_base64_1,
                "mime_type": "image/png",
            },
            {
                "type": "image",
                "base64": image_base64_2,
                "mime_type": "image/png",
            }
        ],
    }],
})

print(result["messages"][-1].content)


print("[bold green]Invoking agent with multimodal input:: image from hybrid...[/bold green]")

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this screenshot?"},
            {"type": "image", "url": r"https://m.media-amazon.com/images/M/MV5BZTNjOWI0ZTAtOGY1OS00ZGU0LWEyOWYtMjhkYjdlYmVjMDk2XkEyXkFqcGc@._V1_QL75_UY281_CR1,0,190,281_.jpg"},
             {
                            "type": "image",
                            "base64": image_base64_1,
                            "mime_type": "image/png",
                        }
        ],
    }],
})


print(result["messages"][-1].content)
