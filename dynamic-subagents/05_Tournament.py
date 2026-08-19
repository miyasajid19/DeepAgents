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

subagent_model = ChatOpenAI(
    model_name="Minimax-M2.7",
    api_key=os.environ.get("MINIMAX_API_KEY"),
    base_url=os.environ.get("MINIMAX_BASE_URL"),
)

# Virtual filesystem scoped to ./dynamic-subagents
backend = FilesystemBackend(root_dir="./dynamic-subagents", virtual_mode=True)

# Three joke generators — each independently produces a joke for the same topic.
# None see the others' work.
joker_lame = {
    "name": "joker-lame",
    "description": "Tells a single lame one-liner joke — groan-worthy, never funny",
    "system_prompt": (
        "You are a comedy specialist that exclusively tells lame, unfunny jokes. "
        "Always reply with exactly ONE short, groan-worthy joke. "
        "Never be clever or genuinely funny."
    ),
    "model": subagent_model,
}

joker_pun = {
    "name": "joker-pun",
    "description": "Tells a corny pun-based joke — dad-joke energy",
    "system_prompt": (
        "You are a pun specialist that exclusively tells corny dad-joke puns. "
        "Always reply with exactly ONE pun on the requested topic. "
        "Lean into wordplay even when it hurts."
    ),
    "model": subagent_model,
}

joker_knock = {
    "name": "joker-knock",
    "description": "Tells a classic knock-knock joke",
    "system_prompt": (
        "You are a knock-knock joke specialist. "
        "Always reply with exactly ONE knock-knock joke on the requested topic. "
        "Use the classic setup-punchline format."
    ),
    "model": subagent_model,
}

# Judge: INVERTED tournament — the lowest score wins, anything too funny
# gets filtered out in code before a winner is chosen.
judge = {
    "name": "joke-judge",
    "description": "Scores each joke on an inverted rubric (lowness wins) and filters to the lamest",
    "system_prompt": (
        "You are an impartial joke judge running an INVERTED tournament. "
        "Three jokers (joker-lame, joker-pun, joker-knock) have each produced "
        "an independent joke for the same topic.\n\n"
        "Score each joke on a 0–10 rubric where HIGHER = FUNNIER, across:\n"
        "  - wit, originality, groan factor, delivery\n"
        "Use the code interpreter to compute the totals in Python.\n\n"
        "FILTER IN CODE: drop any joke scoring 7 or higher — those are too good. "
        "Among the remaining jokes, the LOWEST scorer wins.\n\n"
        "Output: the winner's joke verbatim, the full score table, and a "
        "one-line explanation of why the lame joke reigns."
    ),
    "model": subagent_model,
}

agent = create_deep_agent(
    model=model,
    subagents=[joker_lame, joker_pun, joker_knock, judge],
    middleware=[CodeInterpreterMiddleware()],
    backend=backend,
)


if __name__ == "__main__":
    print("====" * 10)
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                "Run a tournament of jokes on the topic: 'a programmer walks "
                "into a bar'. Dispatch joker-lame, joker-pun, and joker-knock "
                "in parallel, then have joke-judge score, filter, and crown "
                "the lamest joke as the winner."
            )
        }]
    })
    print(result)
