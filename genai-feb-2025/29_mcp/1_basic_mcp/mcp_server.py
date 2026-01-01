# mcp dev mcp_server.py:mcp_server
from mcp.server.fastmcp import FastMCP

mcp_server = FastMCP(name="Greet MCP Server")


@mcp_server.tool()
async def greet(name: str) -> str:
    """
    Greet the user with the given name.

    Args:
        name (str): The name of the user to greet

    Returns:
        (str): The final greeting message
    """
    return f"Hello {name}. Welcome to MCP World!"


if __name__ == "__main__":
    mcp_server.run(transport="stdio")
