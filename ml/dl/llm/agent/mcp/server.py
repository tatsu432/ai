# server.py
from typing import Optional
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DemoTools")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b

@mcp.tool()
def shout(text: str) -> str:
    """Uppercase the input text."""
    return text.upper()

@mcp.tool()
async def fetch_text(url: str, timeout_s: Optional[float] = 5.0) -> str:
    """Fetch raw text from a URL (HTTP GET)."""
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.get(url)
        r.raise_for_status()
        # keep it short for demo
        return r.text[:600]

if __name__ == "__main__":
    # stdio transport is perfect for local development & for LangGraph MCP adapter
    mcp.run(transport="stdio")
