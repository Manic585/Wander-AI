from tools.tavily_tool import tavily_search

# from mcp_client_test import get_all_tools, tavily_mcp_search
from mcp_client import get_all_tools
import asyncio

# def run_tavily_smoke():
#     # Quick manual smoke test for the Tavily search wrapper.
#     result = tavily_search("IMAX Theatres in India")
#     print(result)


# def run_travel_agent_smoke():
#     from backend import run_travel_agent

#     result = run_travel_agent(
#         user_input="4 day budget trip from India to Dubai",
#         thread_id="test",
#     )
#     print(result)


if __name__ == "__main__":
    # run_tavily_smoke()
    asyncio.run(get_all_tools())
    # asyncio.run(
    #     tavily_mcp_search("Top 10 Tamil films that got released in the year 2023")
    # )
