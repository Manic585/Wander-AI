# Wander AI Travel Planner

A FastAPI and LangGraph travel planning app that builds a complete trip plan
from a user request. The app combines flight search, hotel search, itinerary
generation, and final travel recommendations behind a simple blue-and-white web
UI.

## Features

- FastAPI web app with Jinja templates and static frontend assets.
- `/api/travel` endpoint for generating travel plans.
- LangGraph workflow with flight, hotel, itinerary, and final response agents.
- Groq chat model integration through LangChain.
- Tavily hotel/web search integration.
- Aviationstack flight lookup integration.
- PostgreSQL checkpointing through `langgraph-checkpoint-postgres`.
- Markdown response rendering in the browser with local Marked and DOMPurify
  vendor bundles.
- Defensive cleanup for model `<think>` output and incomplete responses.
- Dockerfile for container deployment.

## Requirements

- Python 3.14
- `uv` for local dependency management, or `pip`
- PostgreSQL database URL for LangGraph checkpointing
- API keys for the providers used by the agent

## Setup

Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install dependencies:

```bash
uv sync
```

If you are not using `uv`, install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Environment

Create a local `.env` file for secrets and runtime configuration. The file is
ignored by Git and by Docker builds.

Required values:

```bash
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
DATABASE_URL=postgresql://user:password@host:5432/database
```

Optional LangSmith tracing values:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=your_project_name
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

The backend loads `.env` with `override=True`, so updates to local environment
values are picked up when the app process restarts.

## Run Locally

Start the FastAPI app:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open the UI:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

## API

Generate a travel plan:

```bash
curl -X POST http://127.0.0.1:8000/api/travel \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Plan a 4 day budget trip from India to Dubai",
    "thread_id": "optional-thread-id"
  }'
```

Response fields:

- `thread_id`: conversation/checkpoint thread id.
- `answer`: final travel plan.
- `flight_results`: raw flight agent output.
- `hotel_results`: raw hotel search output.
- `itinerary`: itinerary agent output.
- `llm_calls`: agent step counter.

## Agent Flow

1. The browser submits the trip request to `POST /api/travel`.
2. FastAPI validates the request and calls `run_travel_agent`.
3. The LangGraph workflow runs in this order:
   `flight_agent -> hotel_agent -> itinerary_agent -> final_agent`.
4. Flight and hotel tool outputs are preserved for the UI, while prompt inputs
   are trimmed to stay under provider request limits.
5. Model responses are cleaned with `response_text.py` before being returned.
6. The frontend renders Markdown into formatted headings, lists, links, and
   tables.

## Docker

Build the image:

```bash
docker build -t wander-ai-travel .
```

Run the container with your local `.env` file:

```bash
docker run --env-file .env -p 8000:8000 wander-ai-travel
```

The Dockerfile uses `${PORT:-8000}`, so deployment platforms can provide a
`PORT` environment variable automatically.

## Tests And Checks

Run the regression tests:

```bash
python -m unittest discover -s tests -v
```

Run syntax and whitespace checks:

```bash
python -m py_compile app.py backend.py response_text.py tools/tavily_tool.py tools/flight_tool.py
node --check static/script.js
git diff --check
```

## Project Structure

```text
.
├── app.py
├── backend.py
├── response_text.py
├── main.py
├── Dockerfile
├── .dockerignore
├── tools/
│   ├── flight_tool.py
│   ├── tavily_tool.py
│   └── data/
├── templates/
│   └── index.html
├── static/
│   ├── script.js
│   ├── style.css
│   └── vendor/
├── tests/
│   └── test_response_display.py
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## Notes

- Keep `.env`, virtual environments, caches, and local artifacts out of Git.
- Commit `uv.lock` so dependency resolution stays reproducible.
- Restart the FastAPI process after changing `.env` values.
- Browser vendor files under `static/vendor/` are committed intentionally so
  the UI does not depend on a CDN.
