# Multi Agent System

Python project scaffold for experimenting with multi-agent workflows using
LangGraph, LangChain, FastAPI, Tavily, Groq, and PostgreSQL checkpointing.

## Requirements

- Python 3.14
- `uv` for dependency management

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

Create a local `.env` file for secrets and runtime configuration. This file is
ignored by Git.

Common values for this project may include:

```bash
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
DATABASE_URL=postgresql://user:password@localhost:5432/database
```

Add only the variables needed by the code you are running.

## Run

Run the current Python entry point:

```bash
python main.py
```

The FastAPI/frontend files are present as scaffolding and can be wired up as
the application grows.

## Project Structure

```text
.
├── main.py
├── app.py
├── backend.py
├── tools/
│   ├── flight_tool.py
│   └── tavily_tool.py
├── templates/
│   └── index.html
├── static/
│   ├── script.js
│   └── style.css
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## Notes

- Keep `.env` and virtual environments out of Git.
- Commit `uv.lock` so dependency resolution stays reproducible.
- Update this README as the agent workflow, API routes, and UI are implemented.
