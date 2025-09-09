# agent.py
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

# 1) Connect to one or more MCP servers.
# Here we spawn server.py via stdio.
mcp_client = MultiServerMCPClient(
    {
        "demo": {
            "command": "python",
            "args": ["server.py"],        # use absolute path if running from elsewhere
            "transport": "stdio",
        }
    }
)

async def main():
    # 2) Discover tools from the MCP server(s)
    tools = await mcp_client.get_tools()

    # 3) Choose a chat model you have credentials for.
    #    You can swap providers; LangChain normalizes the interface.
    #    Examples: "openai:gpt-4o-mini", "anthropic:claude-3-7-sonnet-latest", etc.
    model = init_chat_model("openai:gpt-4o-mini")  # set OPENAI_API_KEY env var

    # 4) Build a ReAct-style agent with those tools
    agent = create_react_agent(model, tools)

    # 5) Try the tools
    print("\n== add tool ==")
    r1 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Use the add tool to compute 7 + 35."}]}
    )
    print(r1["messages"][-1].content)

    print("\n== shout tool ==")
    r2 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Call the shout tool on 'agentic ai rules'."}]}
    )
    print(r2["messages"][-1].content)

    print("\n== fetch_text tool ==")
    r3 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Fetch text from https://example.com and summarize it."}]}
    )
    print(r3["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
