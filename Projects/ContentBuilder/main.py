import os
from pathlib import Path
from typing import Literal

import yaml
from langchain.tools import tool

EXAMPLE_DIR = Path(__file__).parent


@tool
def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news"] = "general",
) -> dict:
    """Search the web for current information.

    Args:
        query: The search query (be specific and detailed)
        max_results: Number of results to return (default: 5)
        topic: "general" for most queries, "news" for current events

    Returns:
        Search results with titles, URLs, and content excerpts.
    """
    try:
        from tavily import TavilyClient

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return {"error": "TAVILY_API_KEY not set"}

        client = TavilyClient(api_key=api_key)
        return client.search(query, max_results=max_results, topic=topic)
    except Exception as e:
        return {"error": f"Search failed: {e}"}


@tool
def generate_cover(prompt: str, slug: str) -> str:
    """Generate a cover image for a blog post.

    Args:
        prompt: Detailed description of the image to generate.
        slug: Blog post slug. Image saves to blogs/<slug>/hero.png
    """
    try:
        from google import genai

        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
        )

        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                output_path = EXAMPLE_DIR / "blogs" / slug / "hero.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(output_path))
                return f"Image saved to {output_path}"

        return "No image generated"
    except Exception as e:
        return f"Error: {e}"


@tool
def generate_social_image(prompt: str, platform: str, slug: str) -> str:
    """Generate an image for a social media post.

    Args:
        prompt: Detailed description of the image to generate.
        platform: Either "linkedin" or "tweets"
        slug: Post slug. Image saves to <platform>/<slug>/image.png
    """
    try:
        from google import genai

        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
        )

        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                output_path = EXAMPLE_DIR / platform / slug / "image.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(output_path))
                return f"Image saved to {output_path}"

        return r"Image saved to 'F:\anime\anime\1.jpg'"
    except Exception as e:
        return f"Error: {e}"


def load_subagents(config_path: Path, default_model=None) -> list:
    """Load subagent definitions from YAML and wire up tools.

    Unlike `memory` and `skills`, deep agents do not load subagents from files by default.
    This helper externalizes configuration so you can edit YAML without changing Python code.

    If a subagent spec omits `model:`, it falls back to `default_model` (typically the
    parent agent's already-configured chat model). This keeps every subagent on the
    same provider without needing per-subagent credentials in YAML.
    """
    available_tools = {
        "web_search": web_search,
    }

    with open(config_path) as f:
        config = yaml.safe_load(f)

    subagents = []
    for name, spec in config.items():
        subagent = {
            "name": name,
            "description": spec["description"],
            "system_prompt": spec["system_prompt"],
        }
        # Prefer explicit YAML model; otherwise inherit the parent's model instance
        # so we don't accidentally hit a different provider (e.g., anthropic).
        if "model" in spec:
            subagent["model"] = spec["model"]
        elif default_model is not None:
            subagent["model"] = default_model
        if "tools" in spec:
            subagent["tools"] = [available_tools[t] for t in spec["tools"]]
        subagents.append(subagent)

    return subagents


from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
model=ChatOpenAI(
    model_name="MiniMax-M3",
    api_key=os.environ.get("MINIMAX_API_KEY"),
    base_url=os.environ.get("MINIMAX_BASE_URL")
)
def create_content_writer():
    """Create a content writer agent configured by filesystem files."""
    return create_deep_agent(
        model=model,
        memory=["./AGENTS.md"],
        skills=["./skills/"],
        tools=[generate_cover, generate_social_image],
        subagents=load_subagents(EXAMPLE_DIR / "subagents.yaml", default_model=model),
        backend=FilesystemBackend(root_dir=EXAMPLE_DIR),
    )
    



import sys

from langchain.messages import HumanMessage

if __name__ == "__main__":
    task = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Write a blog post about sajid miya like sajidmiya.tech, aiengineer.sajidmiya.tech, savira.sajidmiya.tech, synapselearn.sajidmiya.tech, nexhire.sajidmiya.tech and sajidmiya.com.np"
    )

    agent = create_content_writer()
    result = agent.invoke(
        {"messages": [HumanMessage(content=task)]},
        config={"configurable": {"thread_id": "content-builder-demo"}},
    )

    for msg in result.get("messages", []):
        if hasattr(msg, "content") and msg.content:
            print(msg.content)