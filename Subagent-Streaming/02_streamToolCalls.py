from deepagents import CompiledSubAgent, create_deep_agent
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import os

try:
	from tavily import TavilyClient
except ImportError:
	TavilyClient = None


load_dotenv()


def internet_search(query: str) -> str:
	"""Run a web search."""
	if TavilyClient is None:
		return "tavily package is not installed. Install it to enable live web search."

	if not os.getenv("TAVILY_API_KEY"):
		return "TAVILY_API_KEY is missing. Unable to run live web search."

	tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
	response = tavily_client.search(query=query, max_results=5)
	results = response.get("results", [])

	if not results:
		return "No web results returned."

	lines = []
	for idx, item in enumerate(results, start=1):
		title = item.get("title", "Untitled")
		url = item.get("url", "")
		content = item.get("content", "")
		lines.append(f"{idx}. {title}\nURL: {url}\n{content}")

	return "\n\n".join(lines)


model = ChatOpenAI(
	model_name="MiniMax-M3",
	api_key=os.getenv("MINIMAX_API_KEY"),
	base_url=os.getenv("MINIMAX_BASE_URL"),
)

research_subagent = {
	"name": "research-agent",
	"description": "Used to research in-depth questions using web search.",
	"system_prompt": "You are a thorough research specialist.",
	"tools": [internet_search],
	"model": model,
}

coding_agent_graph = create_agent(
	model=model,
	tools=[],
	system_prompt="You are a coding expert.",
)

coding_subagent = CompiledSubAgent(
	name="coding-agent",
	description="Used to write and debug code.",
	runnable=coding_agent_graph,
)

joke_subagent = {
	"name": "joke-agent",
	"description": "Used to tell jokes and entertain.",
	"system_prompt": "You are a comedian.",
	"tools": [],
	"model": model,
}

agent = create_deep_agent(
	model=model,
	subagents=[research_subagent, coding_subagent, joke_subagent],
)


if __name__ == "__main__":
	input = {
		"messages": [
			{
				"role": "user",
				"content": "Use tools via subagents and show tool-call streaming output.",
			}
		]
	}

	stream = agent.stream_events(input, version="v3")

	coordinator_tool_names: list[str] = []
	for call in stream.tool_calls:
		print("[coordinator tool]", call.tool_name, call.input)
		print(call.completed, call.error)
		coordinator_tool_names.append(call.tool_name)

	for subagent in stream.subagents:
		for call in subagent.tool_calls:
			print(f"[{subagent.name} tool]", call.tool_name, call.input)
			for delta in call.output_deltas:
				print(delta, end="", flush=True)

			if call.completed and call.error is None:
				print(call.output)
			elif call.error is not None:
				print(call.error)

	print("coordinator tools:", coordinator_tool_names)
