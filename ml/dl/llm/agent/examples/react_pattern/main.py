import os
import json
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from duckduckgo_search import DDGS

try:
	from openai import OpenAI
except Exception:
	OpenAI = None

load_dotenv()
console = Console()

# Simple tools for ReAct

def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
	with DDGS() as ddg:
		return list(ddg.text(query, max_results=max_results))


def react_loop(question: str, model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"), max_steps: int = 6) -> str:
	if OpenAI is None:
		raise RuntimeError("openai package not available")
	client = OpenAI()

	prompt_preamble = (
		"You are an agent that follows the ReAct pattern.\n"
		"At each step, you will think about what to do, then choose one tool to act with, or finish.\n"
		"Respond STRICTLY in JSON with keys: thought, action, action_input, final.\n"
		"Tools: web_search(query), finish(answer).\n"
	)

	trajectory: List[Dict[str, Any]] = []

	for step in range(max_steps):
		messages = [
			{"role": "system", "content": prompt_preamble},
			{"role": "user", "content": json.dumps({"question": question, "trajectory": trajectory})},
		]
		resp = client.chat.completions.create(model=model, messages=messages)
		content = resp.choices[0].message.content or "{}"
		try:
			parsed = json.loads(content)
		except Exception:
			parsed = {"thought": content, "action": "finish", "action_input": None, "final": content}

		thought = parsed.get("thought")
		action = parsed.get("action")
		action_input = parsed.get("action_input")
		final = parsed.get("final")

		console.print(Panel.fit(thought or "(no thought)", title=f"Step {step+1} - Thought"))

		if action == "finish":
			return final or thought or "Done"
		elif action == "web_search":
			results = web_search(action_input or question, max_results=5)
			trajectory.append({"step": step+1, "observation": results[:3]})
			console.print(Panel.fit(json.dumps(results[:3], indent=2)[:2000], title="Observation"))
		else:
			trajectory.append({"step": step+1, "observation": f"Unknown action {action}"})

	return "Reached max steps without finishing."


if __name__ == "__main__":
	q = os.environ.get("QUESTION", "What is the latest about quantum error correction?")
	ans = react_loop(q)
	console.print(Rule("Final Answer"))
	console.print(ans)
