import os
from pathlib import Path
import certifi
from dotenv import load_dotenv
import asyncio
import json

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq

# from tools.flight_tool import search_flights
from response_text import strip_thinking, visible_model_response
from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    get_current_weather,
    get_forecast,
)

FLIGHT_AGENT_PROMPT = """
You are a travel flight expert

User Query : 
{query}

Airport information:
{airport_data}

Airline information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving the route
4. Flight duration
5. Airfare rate
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance
"""


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ API Key is missing. Please add it to your environment variable"
    )

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    max_tokens=4096,
    reasoning_format="hidden",
)

MAX_USER_INPUT_CHARS = 1200
MAX_FLIGHT_PROMPT_CHARS = 3500
MAX_HOTEL_PROMPT_CHARS = 3500
MAX_ITINERARY_PROMPT_CHARS = 5500


def ensure_text(value) -> str:
    if isinstance(value, str):
        return value

    if value is None:
        return ""

    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def trim_for_prompt(value: str, max_chars: int) -> str:
    text = ensure_text(value).strip()

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars].rsplit("\n", 1)[0].strip()
        + "\n\n[Trimmed to keep the LLM request within provider limits.]"
    )


class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    weather_results: str


def flight_agent(state: TravelState):
    query = state["user_query"]

    try:
        airports = asyncio.run(aviation_mcp_call("list_airports", {}))
        airlines = asyncio.run(aviation_mcp_call("list_airlines", {}))

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=trim_for_prompt(airports, MAX_FLIGHT_PROMPT_CHARS),
            airline_data=trim_for_prompt(airlines, MAX_FLIGHT_PROMPT_CHARS),
        )

        response = llm.invoke(
            [
                SystemMessage(content="You are an expert travel flight planner"),
                HumanMessage(content=prompt),
            ]
        )
        flight_data = visible_model_response(response)

    except Exception as e:
        flight_data = f"Flight information unavailable - {str(e)}"

    return {
        "flight_results": flight_data,
        "llm_calls": state.get("llm_calls", 0) + 1,
        "messages": [AIMessage(content="Flight results fetched")],
    }


def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_data = ensure_text(asyncio.run(tavily_mcp_search(query)))

    return {
        "hotel_results": hotel_data,
        "messages": [AIMessage(content="Hotel data fetched")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def weather_agent(state: TravelState):

    city = extract_destination(state["user_query"])

    if city == "NA":
        weather_results = "Weather information unavailable: destination not found."
    else:
        try:
            weather_data = asyncio.run(get_current_weather(city))
            forecast_data = asyncio.run(get_forecast(city))
            weather_results = f"""
            Current Weather: {ensure_text(weather_data)}
            Forecast: {ensure_text(forecast_data)}
            """
        except Exception as exc:
            weather_results = f"Weather information unavailable - {str(exc)}"

    return {
        "weather_results": weather_results.strip(),
        "llm_calls": state.get("llm_calls", 0) + 1,
        "messages": [AIMessage(content="Weather results fetched")],
    }


def itinerary_agent(state: TravelState):
    flight_results = trim_for_prompt(state["flight_results"], MAX_FLIGHT_PROMPT_CHARS)
    hotel_results = trim_for_prompt(state["hotel_results"], MAX_HOTEL_PROMPT_CHARS)
    weather_results = state.get("weather_results", "")

    prompt = f"""
        Create a complete travel itinerary.

        User Query:
        {state["user_query"]}

        Flight Results:
        {flight_results}

        Hotel Results:
        {hotel_results}

        Weather Results:
        {trim_for_prompt(weather_results, MAX_HOTEL_PROMPT_CHARS)}

        Make the itinerary practical, budget-aware, and easy to follow.
        Return only the travel plan in Markdown with ## headings and lists.
        Cover every requested day within 1800 words. Do not include thinking text.
        """

    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ]
    )

    response = response.model_copy(update={"content": visible_model_response(response)})
    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def final_agent(state: TravelState):
    flight_results = trim_for_prompt(state["flight_results"], 1800)
    hotel_results = trim_for_prompt(state["hotel_results"], 1800)
    weather_results = state.get("weather_results", "")
    itinerary = trim_for_prompt(
        strip_thinking(state["itinerary"]), MAX_ITINERARY_PROMPT_CHARS
    )

    prompt = f"""
            Generate the final travel response for the user

            User Query:
            {state["user_query"]}

            Flight Summary Source:
            {flight_results}

            Hotel Summary Source:
            {hotel_results}

            Weather:
            {weather_results}

            Itinerary Results:
            {itinerary}

            Format the final response using these sections.

            1. Trip Summary
            2. Flight Information
            3. Hotel Suggestions
            4. Day by Day itinerary
            5. Estimated Budget
            6. Final Recommendations
            7. Other things to be kept in mind

            Important:
            - Be clear and practical
            - Mention that live flight API may not provide ticket prices if pricing is unavailable
            - Keep the response useful for real travel planning
            - Return only the travel plan in Markdown with ## section headings
            - Cover all seven sections within 1800 words; do not include thinking text
            """
    response = llm.invoke(
        [
            SystemMessage(content="You are a professional tour planning agent"),
            HumanMessage(content=prompt),
        ]
    )

    response = response.model_copy(update={"content": visible_model_response(response)})
    return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}


graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


DATABASE_URL = get_database_url()
conn = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)
checkpointer = PostgresSaver(conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


def run_travel_agent(user_input: str, thread_id: str | None = None):
    user_input = trim_for_prompt(user_input, MAX_USER_INPUT_CHARS)

    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "llm_calls": 0,
        },
        config=config,
    )

    final_response = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_response,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }
