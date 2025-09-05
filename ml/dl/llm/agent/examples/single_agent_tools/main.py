import os
import json
import time
import uuid
import math
from typing import Any, Dict, List, Optional

import click
import requests
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Optional tool deps
from duckduckgo_search import DDGS
import faiss
import numpy as np

try:
	from openai import OpenAI
except Exception:
	OpenAI = None

load_dotenv()
console = Console()

# ---------------------- Schemas ----------------------

class WebSearchArgs(BaseModel):
	query: str = Field(..., min_length=2, description="Search query for the web")
	max_results: int = Field(5, ge=1, le=10)

class CalcArgs(BaseModel):
	expression: str = Field(..., description="Pythonic math expression, safe subset")

class HttpGetArgs(BaseModel):
	url: str
	timeout: int = 10

class AddNoteArgs(BaseModel):
	text: str

# ---------------------- Tool Implementations ----------------------

NOTES: List[str] = []


def tool_web_search(args: WebSearchArgs) -> List[Dict[str, Any]]:
	with DDGS() as ddg:
		results = list(ddg.text(args.query, max_results=args.max_results))
	return results


def tool_calc(args: CalcArgs) -> Dict[str, Any]:
	# Extremely constrained eval
	allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
	allowed_names.update({"__builtins__": {}})
	value = eval(args.expression, allowed_names, {})
	return {"result": value}


def tool_http_get(args: HttpGetArgs) -> Dict[str, Any]:
	r = requests.get(args.url, timeout=args.timeout)
	return {"status": r.status_code, "text": r.text[:2000]}


def tool_add_note(args: AddNoteArgs) -> Dict[str, Any]:
	NOTES.append(args.text)
	return {"ok": True, "count": len(NOTES)}

# ---------------------- Embeddings + Simple RAG ----------------------

EMBED_DIM = 384

try:
	from sentence_transformers import SentenceTransformer
	emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
	embed = lambda x: emb_model.encode(x, normalize_embeddings=True)
	EMBED_DIM = 384
except Exception:
	# Fallback to random (demo only)
	embed = lambda x: np.random.randn(len(x), EMBED_DIM).astype("float32")


class VectorStore:
	def __init__(self):
		self.index = faiss.IndexFlatIP(EMBED_DIM)
		self.texts: List[str] = []

	def add_texts(self, texts: List[str]):
		vecs = embed(texts)
		if not isinstance(vecs, np.ndarray):
			vecs = np.array(vecs)
		vecs = vecs.astype("float32")
		self.index.add(vecs)
		self.texts.extend(texts)

	def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
		q = embed([query]).astype("float32")
		dists, idxs = self.index.search(q, k)
		results = []
		for score, i in zip(dists[0], idxs[0]):
			if i == -1 or i >= len(self.texts):
				continue
			results.append({"text": self.texts[i], "score": float(score)})
		return results

VSTORE = VectorStore()

# seed with notes store contents dynamically

# ---------------------- Tool Registry ----------------------

TOOLS = {
	"web_search": (WebSearchArgs, tool_web_search, "Search the web for current information."),
	"calc": (CalcArgs, tool_calc, "Evaluate a safe math expression using Python math module."),
	"http_get": (HttpGetArgs, tool_http_get, "HTTP GET a URL and return status and snippet."),
	"add_note": (AddNoteArgs, tool_add_note, "Add a note to the session notes store."),
	"vector_search": (
		WebSearchArgs,
		lambda a: VSTORE.search(a.query, a.max_results),
		"Search a local vector store over session-ingested texts.",
	),
}


def list_tools_for_llm() -> List[Dict[str, Any]]:
	return [
		{
			"type": "function",
			"function": {
				"name": name,
				"description": desc,
				"parameters": args.model_json_schema(),
			},
		}
		for name, (args, _impl, desc) in TOOLS.items()
	]


# ---------------------- LLM Orchestrator ----------------------

def call_openai(model: str, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None):
	if OpenAI is None:
		raise RuntimeError("openai package not available")
	client = OpenAI()
	resp = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto")
	return resp


def run_orchestrator(task: str, model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")):
	messages: List[Dict[str, Any]] = [
		{"role": "system", "content": "You are a precise orchestrator. Use tools when helpful. Keep responses concise."},
		{"role": "user", "content": task},
	]
	tools = list_tools_for_llm()

	while True:
		resp = call_openai(model, messages, tools=tools)
		choice = resp.choices[0]
		if choice.message.tool_calls:
			for tc in choice.message.tool_calls:
				name = tc.function.name
				raw_args = json.loads(tc.function.arguments or "{}")
				if name not in TOOLS:
					messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": json.dumps({"error": "unknown tool"})})
					continue
				args_model, impl, _ = TOOLS[name]
				try:
					parsed = args_model(**raw_args)
					result = impl(parsed)
					if name == "add_note":
						VSTORE.add_texts([parsed.text])
					tool_content = json.dumps(result)
				except ValidationError as ve:
					tool_content = json.dumps({"validation_error": ve.errors()})
				messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": tool_content})
			# continue loop to let model see tool result
			continue
		else:
			assistant_msg = choice.message
			messages.append({"role": "assistant", "content": assistant_msg.content})
			return assistant_msg.content


# ---------------------- CLI ----------------------

@click.group()
def cli():
	"""Single-agent with function-calling tools demo."""


@cli.command()
@click.argument("task", nargs=-1)
@click.option("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
def run(task: List[str], model: str):
	task_str = " ".join(task).strip()
	if not task_str:
		console.print("Provide a task, e.g., 'search latest NVIDIA news and add note'.")
		return
	console.print(Panel(f"Model: [bold]{model}[/bold]\nTask: {task_str}"))
	try:
		answer = run_orchestrator(task_str, model=model)
		console.print(Panel.fit(answer or "(no content)", title="Assistant"))
	except Exception as e:
		console.print(f"[red]Error:[/red] {e}")


if __name__ == "__main__":
	cli()
