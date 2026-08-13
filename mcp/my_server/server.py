from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my_server")


@mcp.tool()
def hello(name: str) -> str:
    """Return a greeting message."""
    
    return f"Hello, {name}!"


@mcp.tool()
def add(a: int, b: int) -> int:
    '''
        this function adds two numbers and returns the result
        a: int - first number
        b: int - second number
    '''
    
    return a * b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    '''
        this function multiplies two numbers and returns the result
        a: int - first number
        b: int - second number
    '''
    return a * b


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
    )