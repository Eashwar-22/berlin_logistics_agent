from mcp.server.fastmcp import FastMCP
from agent import run_agent_logic

# Server
mcp = FastMCP("BerlinLogisticsAgent")

@mcp.tool()
def ask_berlin_agent(query: str) -> str:
    """
    Ask anything about Berlin deliveries, weather, or PII.
    The agent will reason, use sub-tools, and return an answer.
    """
    return run_agent_logic(query)

if __name__ == "__main__":
    mcp.run(transport='stdio')