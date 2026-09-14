from tavily import TavilyClient
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

# Reuse one Tavily client for all searches in this module.
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def tavily_search(query: str):
    response = client.search(query=query, max_results=3)
    results = []
    for i, r in enumerate(response["results"], 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "").strip()

        result = f"### {i}. {title}\n\n{url}\n\n{content}"
        results.append(result)

    return "\n\n".join(results)
