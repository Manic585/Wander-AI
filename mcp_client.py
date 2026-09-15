import os
import sys
from pathlib import Path

import certifi
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq

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

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

search_tool = None
current_weather_tool = None
forecast_tool = None
aviation_tools = {}

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    max_tokens=4096,
    reasoning_format="hidden",
)


async def initialize_tavily_tool():
    global search_tool
    tools = await client.get_tools(server_name="tavily")
    search_tool = next(tool for tool in tools if tool.name == "tavily_search")
    return search_tool


async def initialize_aviation_tools():
    global aviation_tools
    tools = await client.get_tools(server_name="aviationstack")
    aviation_tools = {tool.name: tool for tool in tools}
    return aviation_tools


async def initialize_weather_tools():
    global current_weather_tool, forecast_tool

    if not OPENWEATHER_API_KEY:
        raise ValueError(
            "OPENWEATHER_API_KEY is missing. Please add it to your environment."
        )

    tools = await client.get_tools(server_name="weather")
    current_weather_tool = next(
        tool for tool in tools if tool.name == "get_current_weather"
    )
    forecast_tool = next(tool for tool in tools if tool.name == "get_forecast")
    return current_weather_tool, forecast_tool


async def get_all_tools():
    return await client.get_tools()


async def tavily_mcp_search(query: str):
    await initialize_tavily_tool()
    result = await search_tool.ainvoke({"query": query})
    return result


async def aviation_mcp_call(tool_name: str, tool_args: dict):
    await initialize_aviation_tools()

    tool = aviation_tools.get(tool_name)
    if tool is None:
        raise ValueError(f"Unknown aviation MCP tool: {tool_name}")
    result = await tool.ainvoke(tool_args or {})

    return result


async def get_current_weather(city: str):
    if not OPENWEATHER_API_KEY:
        raise ValueError(
            "OPENWEATHER_API_KEY is missing. Please add it to your environment."
        )

    await initialize_weather_tools()
    return await current_weather_tool.ainvoke({"city": city})


async def get_forecast(city: str):
    if not OPENWEATHER_API_KEY:
        raise ValueError(
            "OPENWEATHER_API_KEY is missing. Please add it to your environment."
        )

    await initialize_weather_tools()
    return await forecast_tool.ainvoke({"city": city})


def extract_destination(query: str):
    prompt = f"""
    Extract only the destination from the query 
    {query}

    Return only the destination name. 
    If not found, return NA.
    """

    response = llm.invoke(prompt)

    return response.content.strip()


servers = {
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
    "weather": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(Path(__file__).resolve().parent / "weather_mcp_server.py")],
        "env": {"OPENWEATHER_API_KEY": OPENWEATHER_API_KEY or ""},
    },
}

client = MultiServerMCPClient(servers)
