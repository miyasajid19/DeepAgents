from deepagents import (
    create_deep_agent,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()
from pydantic import BaseModel,Field
from typing import Literal, Optional
class Sentiment(BaseModel):
    comment: str = Field(..., description="The comment to analyze sentiment for.")
    sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="The sentiment of the comment.") 
    confidence: Optional[float] = Field(..., description="The confidence score of the sentiment analysis.")
    
model= ChatOpenAI(
    model_name="MiniMax-M3",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
)
register_harness_profile(
    "openai:MiniMax-M3",
    HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(
            enabled=False
        ),
    ),
)

sentiment_extraction_subagent = {
    "name": "sentiment-extraction-agent",
    "description": "Extracts sentiment from text.",
    "system_prompt": "You are a sentiment analysis expert.",
    "response_format": Sentiment,
    }

agent = create_deep_agent(
    model=model,
    subagents=[sentiment_extraction_subagent],
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "Analyze the sentiment of the following comment: 'I love this product! It's amazing and exceeded my expectations.'"}]})
    from rich import print
    print(result)
    print("[bold green]=================================================[/bold green]")
    print("[red] Agent's Response:[/red]")
    print(result["messages"][-1].content)