from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Wander AI", description="Multi agent travel planner")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class TravelRequest(BaseModel):
    user_input: str
    thread_id: str | None = None


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/travel")
def create_travel_plan(payload: TravelRequest):
    user_input = payload.user_input.strip()
    thread_id = payload.thread_id.strip() if payload.thread_id else None

    if not user_input:
        raise HTTPException(status_code=400, detail="Trip request is required.")

    try:
        from backend import run_travel_agent

        return run_travel_agent(user_input=user_input, thread_id=thread_id)
    except HTTPException:
        raise
    except Exception as exc:
        error_message = str(exc)

        if "request_too_large" in error_message or "Request Entity Too Large" in error_message or "413" in error_message:
            raise HTTPException(
                status_code=413,
                detail=(
                    "The travel request became too large for the LLM provider. "
                    "I trimmed tool results, but try a shorter trip request or use a new Thread ID."
                ),
            ) from exc

        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "AI Travel Planner API is running"}


@app.get("/favicon.ico")
def favicon():
    return JSONResponse(content={})


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
