import os
import json
from typing import Any, Dict, List

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from tenacity import retry, stop_after_attempt, wait_exponential

try:
	from openai import OpenAI
except Exception:
	OpenAI = None

load_dotenv()
console = Console()

# ---------------- Planner ----------------

def plan(task: str, model: str) -> List[Dict[str, Any]]:
	if OpenAI is None:
		raise RuntimeError("openai package not available")
	client = OpenAI()
	messages = [
		{"role": "system", "content": "You are a planner. Decompose the task into numbered, minimal, executable steps as JSON list."},
		{"role": "user", "content": task},
	]
	resp = client.chat.completions.create(model=model, messages=messages)
	content = resp.choices[0].message.content or "[]"
	try:
		return json.loads(content)
	except Exception:
		return [{"step": 1, "action": "Describe high-level approach", "args": {"task": task}}]

# ---------------- Executor ----------------

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
def execute(steps: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
	if OpenAI is None:
		raise RuntimeError("openai package not available")
	client = OpenAI()
	results = []
	for i, step in enumerate(steps, 1):
		messages = [
			{"role": "system", "content": "You are an executor. Perform the step and return JSON with keys: result, logs."},
			{"role": "user", "content": json.dumps({"step": step})},
		]
		resp = client.chat.completions.create(model=model, messages=messages)
		content = resp.choices[0].message.content or "{}"
		try:
			parsed = json.loads(content)
		except Exception:
			parsed = {"result": content, "logs": "unstructured"}
		results.append({"step_index": i, "step": step, "output": parsed})
	return results

# ---------------- Verifier ----------------

def verify(task: str, plan_steps: List[Dict[str, Any]], exec_results: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
	if OpenAI is None:
		raise RuntimeError("openai package not available")
	client = OpenAI()
	messages = [
		{"role": "system", "content": "You are a verifier. Check if outputs satisfy the task. Return JSON with keys: ok, issues, fixes, final_answer."},
		{"role": "user", "content": json.dumps({"task": task, "plan": plan_steps, "results": exec_results})},
	]
	resp = client.chat.completions.create(model=model, messages=messages)
	content = resp.choices[0].message.content or "{}"
	try:
		return json.loads(content)
	except Exception:
		return {"ok": False, "issues": ["unstructured"], "fixes": [], "final_answer": content}


if __name__ == "__main__":
	model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
	task = os.getenv("TASK", "Write an outline for a blog about vector databases and compare FAISS vs Chroma.")
	plan_steps = plan(task, model)
	console.print(Panel.fit(json.dumps(plan_steps, indent=2), title="Plan"))
	exec_results = execute(plan_steps, model)
	console.print(Panel.fit(json.dumps(exec_results, indent=2)[:2000], title="Executor Outputs"))
	verdict = verify(task, plan_steps, exec_results, model)
	console.print(Panel.fit(json.dumps(verdict, indent=2), title="Verifier Verdict"))
