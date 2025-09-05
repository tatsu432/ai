import os
import json
from typing import Any, Dict, List

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# We'll default to OpenAI via CrewAI for simplicity
from crewai import Agent, Task, Crew, Process

load_dotenv()
console = Console()

# Define specialist agents
researcher = Agent(
	role="Senior Researcher",
	goal="Find current, credible information and summarize concisely",
	backstory="You are rigorous and cite sources.",
	verbose=True,
)

writer = Agent(
	role="Technical Writer",
	goal="Explain complex topics clearly for a smart audience",
	backstory="You transform research into crisp writing with structure.",
	verbose=True,
)

critic = Agent(
	role="Critical Reviewer",
	goal="Catch errors, improve clarity, and ensure evidence-based claims",
	backstory="You review and suggest edits.",
	verbose=True,
)


def run_crew(topic: str) -> str:
	research_task = Task(
		description=f"Research the topic: {topic}. Provide a bullet list with source links.",
		agent=researcher,
	)
	writing_task = Task(
		description="Turn the research into a 5-paragraph summary with headings.",
		agent=writer,
		requires=[research_task],
	)
	review_task = Task(
		description="Critically review the summary, fixing inaccuracies and improving clarity.",
		agent=critic,
		requires=[writing_task],
	)

	crew = Crew(agents=[researcher, writer, critic], tasks=[research_task, writing_task, review_task], process=Process.sequential)
	result = crew.kickoff()
	return result


if __name__ == "__main__":
	topic = os.getenv("TOPIC", "State of open-source small LLMs in 2025")
	result = run_crew(topic)
	console.print(Panel.fit(result, title="Crew Output"))
