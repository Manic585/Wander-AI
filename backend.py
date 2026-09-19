import os
from pathlib import Path
import certifi
from dotenv import load_dotenv
import asyncio
import json

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated, Any
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
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

KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]

ROUTE_MAP = {
    "guardrail_blocked_agent": "guardrail_blocked_agent",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}
MAX_USER_INPUT_CHARS = 1200
MAX_FLIGHT_PROMPT_CHARS = 3500
MAX_HOTEL_PROMPT_CHARS = 3500
MAX_ITINERARY_PROMPT_CHARS = 5500

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

GUARDRAIL_PROMPT = """
Determine whether the following request belongs to travel planning or travel inforation. Valid requests can include destinations, flights, hotels, weather, budgets, visas, transportation, sightseeing, food, packing, itineraries etc

Block clearly unrelated requests having harmful or illegal instructions. Don't block a valid travel request merely because some fields are missing

Return strict JSON only:
{{
"allowed" : True,
"reason: ""
}}

User request : {query}
"""

SUPERVISOR_PROMPT = """
You are the supervisor agent of a multi agent travel planning system.
Choose only the specialist agents needed for the request. 

Available agents :
- flight_agent : flights, airports, airlines, airfares, routes, booking advice
- hotel_agent : hotels, accomodation, stay, places, neighborhood
- weather_agent : weather, climate, season, rainy, sunny, forecast, packing advice
- budget_agent : budget, cost, affordability, money, price, feasibility
- itinerary_agent : itinerary, travel plan - creates the integrated travel plan and must always be included.

Return strict JSON using this schema :
{{
    "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
    "trip_constraints": {{
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": []
    }},
    "supervisor_reasoning": ""
}}

User request : {query}
"""

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ API Key is missing. Please add it to your environment variable"
    )

llm = ChatGroq(
    model="openai/gpt-oss-safeguard-20b",
    api_key=GROQ_API_KEY,
    max_tokens=4096,
    reasoning_format="hidden",
)


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

    # Supervisor and guardrail state
    guardrail_allowed: bool  # Request is allowed or not
    guardrail_reason: str  # Reason for rejection
    selected_agents: list[str]  # Agents for the query
    trip_constraints: dict[str, Any]  # Constraints found in the query
    supervisor_reasoning: str  # Why supervisor agent picks the specific agents

    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    budget_results: str
    approval_request: str  # When user approves
    isApproved: bool  # True or false
    human_feedback: str  # If rejected, what is the feedback
    final_response: str  # Final agent response

    llm_calls: int


def selected_agent(state: TravelState) -> str:
    selected_agents = state.get("selected_agents", [])

    return next(
        (agent for agent in AGENT_ORDER if agent in selected_agents), "itinerary_agent"
    )


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed"):
        return "guardrail_blocked_agent"

    return selected_agent(state)


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected_agents = state.get("selected_agents")
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected_agents:
                return next_agent
        return "itinerary_agent"

    return route


# Helper functions


def llm_invoke(user_message: str, system_message: str):
    response = llm.invoke(
        [SystemMessage(content=system_message), HumanMessage(content=user_message)]
    )
    return str(response.content)


def json_from_llm(text: str) -> dict[str, any]:
    """Extract the first complete JSON object from the model"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No complete JSON object found")
    return json.loads(text[start : end + 1])


def empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    # Guardrail for the request
    try:
        prompt = GUARDRAIL_PROMPT.format(query=query)
        response = llm_invoke(
            prompt,
            "You are the input guardrail for a travel planning application. Return strict JSON only",
        )
        json_from_llm(response)

        guardrail_result = json_from_llm(response)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as ex:
        print(f"Guardrail rejected the request {ex}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request"
    if not allowed:
        guardrail_reason = "Wander AI can only help with travel planning requests. Please provide your query for valid travel related planning"
        return {
            "messages": [
                AIMessage(content=f"Guardrail blocked the request {guardrail_reason}")
            ],
            "guardrail_allowed": False,
            "guardrail_reason": guardrail_reason,
            "selected_agents": [],
            "trip_constraints": empty_constraints(),
            "supervisor_reasoning": "",
            "llm_calls": llm_calls,
        }

    try:
        prompt = SUPERVISOR_PROMPT.format(query=query)
        response = llm_invoke(
            prompt,
            "You are a supervisor agent in a multi agent travel planner. Return strict JSON only",
        )
        supervisor_result = json_from_llm(response)
        returned_agents = supervisor_result.get("selected_agents", [])
        selected_agents = [agent for agent in AGENT_ORDER if agent in returned_agents]
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")
        trip_constraints = supervisor_agent.get("trip_constraints", {})
        supervisor_reasoning = str(
            supervisor_agent.get("supervisor_reasoning", "")
        ).strip()
        llm_calls += 1

    except Exception as ex:
        print(f"Supervisor agent fallback used {ex}")
        selected_agents = AGENT_ORDER.copy()
        trip_constraints = empty_constraints()
        supervisor_reasoning = """
        Supervisor agent failed. So the original travel workflow comprising all the agents is executed
        """

    return {
        "selected_agents": selected_agents,
        "trip_constraints": trip_constraints,
        "supervisor_reasoning": supervisor_reasoning,
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "messages": [
            AIMessage(content="Supervisor picked the specific agents for the query")
        ],
        "llm_calls": llm_calls,
    }


def guardrail_blocked_agent(state: TravelState):
    reason = (
        state.get("guardrail_reason")
        or "Wander AI can only help with travel planning requests. Please provide your query for valid travel related planning"
    )
    return {
        "guardrail_allowed": False,
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


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


def budget_agent(state: TravelState):
    prompt = f"""
    Analyse whether the trip is realistic for the user's budget

    User query : 
    {state["user_query"]}

    Trip constraints : 
    {state["trip_constraints"]}

    Flight Results:
    {state.get("flight_results", "")}

    Hotel Results:
    {state.get("hotel_results", "")}

    Weather Results:
    {state.get("weather_results", "")}

    Return 
    1. Estimated Cost categories
    2. Budget risk areas
    3. Money saving suggestions
    4. Overall feasibility

    If exact live prices are unavailable, clearly label them as approximate. 

    """
    response = llm_invoke(prompt, "You are a practical travel budget analyst")

    return {
        "budget_results": response,
        "messages": [AIMessage("Budget results generated successfully")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def itinerary_agent(state: TravelState):
    flight_results = trim_for_prompt(state["flight_results"], MAX_FLIGHT_PROMPT_CHARS)
    hotel_results = trim_for_prompt(state["hotel_results"], MAX_HOTEL_PROMPT_CHARS)
    weather_results = state.get("weather_results", "")
    budget_results = state.get("budget_results", "")
    trip_constraints = state.get("trip_constraints", "")

    prompt = f"""
        Create a complete travel itinerary.

        User Query:
        {state["user_query"]}

        Trip Constraints:
        {trip_constraints}

        Flight Results:
        {flight_results}

        Hotel Results:
        {hotel_results}

        Weather Results:
        {trim_for_prompt(weather_results, MAX_HOTEL_PROMPT_CHARS)}

        Budget Results:
        {budget_results}

        Make the itinerary practical, budget-aware, and easy to follow.
        Create a clear draft that is ready for human review
        """

    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ]
    )

    approval_request = """Please review the generated draft itinerary. Approve it to create the final polished itinerary plan or provide the feedback for revision"""

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def human_approval_agent(state: TravelState):
    review = interrupt(
        {
            "question": "Do you want to approve this itinerary?",
            "draft_itinerary": state.get("itinerary"),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning"),
            "approval_request": state.get("approval_request", ""),
            "expected_response": {"approved": True, "feedback": "Optional feedback"},
        }
    )

    isApproved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "isApproved": isApproved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed")],
    }


def final_agent(state: TravelState):
    if state.get("isApproved"):
        human_review = "Human has approved the itinerary request generated"
    else:
        human_review = f"Human has rejected the itinerary request generated. Refer the human feedback {state.get('human_feedback')} and improve the draft before finalising"

    flight_results = trim_for_prompt(state["flight_results"], 1800)
    hotel_results = trim_for_prompt(state["hotel_results"], 1800)
    weather_results = state.get("weather_results", "")
    budget_results = state.get("budget_results", "")
    itinerary = trim_for_prompt(
        strip_thinking(state["itinerary"]), MAX_ITINERARY_PROMPT_CHARS
    )

    prompt = f"""
            Generate the final travel response for the user.

            Human review:
            {human_review}

            User Query:
            {state["user_query"]}

            Supervisor constraints:
            {state["trip_constraints"]}

            Flight Summary Source:
            {flight_results}

            Hotel Summary Source:
            {hotel_results}

            Weather:
            {weather_results}

            Budget:
            {budget_results}

            Itinerary Results:
            {itinerary}

            Format the final response using these sections.

            1. Trip Summary
            2. Flight Information
            3. Hotel Suggestions
            4. Weather Information
            5. Day by Day itinerary
            6. Estimated Budget
            7. Final Recommendations
            8. Other things to be kept in mind

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

    return {
        "response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


graph = StateGraph(TravelState)

graph.add_node("supervisor_agent", supervisor_agent)
graph.add_node("guardrail_blocked_agent", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval_agent", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor_agent")
graph.add_conditional_edges("supervisor_agent", route_from_supervisor, ROUTE_MAP)
graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges("hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)
graph.add_edge("itinerary_agent", "human_approval_agent")
graph.add_edge("human_approval_agent", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked_agent", END)


DATABASE_URL = get_database_url()
conn = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)
checkpointer = PostgresSaver(conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


def get_interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first LangGraph interrupt payload, if the workflow is paused."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None

    interrupt_item = (
        interrupts[0] if isinstance(interrupts, (list, tuple)) else interrupts
    )
    payload = getattr(interrupt_item, "value", interrupt_item)

    if isinstance(payload, dict):
        return payload

    return {"value": payload}


def _travel_response(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    messages = result.get("messages", [])
    answer = messages[-1].content if messages else result.get("final_response", "")
    interrupt_payload = get_interrupt_payload(result)

    return {
        "thread_id": thread_id,
        "answer": answer,
        "interrupt_payload": interrupt_payload,
        "requires_approval": interrupt_payload is not None,
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", empty_constraints()),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": result.get("itinerary", ""),
        "approval_request": result.get("approval_request", ""),
        "isApproved": result.get("isApproved", False),
        "human_feedback": result.get("human_feedback", ""),
        "final_response": result.get("final_response", answer),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    user_input = trim_for_prompt(user_input, MAX_USER_INPUT_CHARS)

    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "budget_results": "",
            "approval_request": "",
            "isApproved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _travel_response(result, thread_id)


def resume_travel_agent(thread_id: str, approved: bool, human_feedback: str):
    """Resume a paused travel workflow with the human review response."""
    if not thread_id or not thread_id.strip():
        raise ValueError("thread_id is required to resume a travel workflow")

    config = {"configurable": {"thread_id": thread_id.strip()}}
    result = travel_graph.invoke(
        Command(resume={"approved": approved, "feedback": human_feedback}),
        config=config,
    )

    return _travel_response(result, thread_id.strip())
