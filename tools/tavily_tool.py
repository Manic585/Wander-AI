from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

# Reuse one Tavily client for all searches in this module.
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def tavily_search(query: str):
    response = client.search(query=query, max_results=5)
    results = []
    for i, r in enumerate(response["results"], 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "").strip()

        # Keep each search hit short enough to fit cleanly in an agent response.
        if len(content) > 300:
            content = content[:300].rsplit(" ", 1)[0] + "..."

        result = f"{i}. **{title}** \n {url} \n {content}"
        results.append(result)

    return "\n\n".join(results)
