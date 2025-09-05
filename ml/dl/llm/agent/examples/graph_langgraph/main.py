import os
from typing import Dict, Any

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
console = Console()

# Define state
class State(dict):
	pass

# Nodes

def plan_node(state: State) -> State:
	model = init_chat_model(os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
	resp = model.invoke([
		HumanMessage(content=f"Create a short plan for: {state['task']}")
	])
	state["plan"] = resp.content
	return state


def execute_node(state: State) -> State:
	model = init_chat_model(os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
	resp = model.invoke([
		HumanMessage(content=f"Execute the plan succinctly:\n{state['plan']}")
	])
	state["output"] = resp.content
	return state


def review_node(state: State) -> State:
	model = init_chat_model(os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
	resp = model.invoke([
		HumanMessage(content=f"Review the output for quality and correctness. Provide fixes if needed. Output:\n{state['output']}")
	])
	state["review"] = resp.content
	return state


# Graph
workflow = StateGraph(State)
workflow.add_node("plan", plan_node)
workflow.add_node("execute", execute_node)
workflow.add_node("review", review_node)
workflow.set_entry_point("plan")
workflow.add_edge("plan", "execute")
workflow.add_edge("execute", "review")
workflow.add_edge("review", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
	task = os.getenv("TASK", "Draft a 3-bullet product update for our AI agent release.")
	initial_state: State = {"task": task}
	final_state = app.invoke(initial_state)
	console.print(Panel.fit(final_state.get("review", "(no review)"), title="Final Review"))
