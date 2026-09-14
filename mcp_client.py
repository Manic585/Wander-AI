import os
from pathlib import Path

import certifi
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing. Please add it to your environment.")

AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
if not AVIATIONSTACK_API_KEY:
    raise ValueError(
        "AVIATIONSTACK_API_KEY is missing. Please add it to your environment."
    )

search_tool = None
aviation_tool = {}


async def mcp_initialize():
    global search_tool

    tools = await client.get_tools()

    for tool in tools:
        print(tool.name)

    search_tool = next(tool for tool in tools if tool.name == "tavily_search")


async def tavily_mcp_search(query: str):
    await mcp_initialize()
    result = await search_tool.ainvoke({"query": query})
    return result


async def aviation_mcp_call(tool_name: str, tool_args: dict):
    await mcp_initialize()

    tools = await client.get_tools()

    tool = next(tool for tool in tools if tool.name == tool_name)

    result = await tool.ainvoke(tool_args or {})

    return result


client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
        },
        "aviationstack": {
            "transport": "stdio",
            "command": "uvx",
            "args": ["aviationstack-mcp"],
            "env": {"AVIATION_STACK_API_KEY": AVIATIONSTACK_API_KEY},
        },
    }
)
