import unittest
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from response_text import strip_thinking, visible_model_response


class ResponseTextTests(unittest.TestCase):
    def test_removes_reasoning_but_preserves_complete_answer(self):
        answer = "## Trip Summary\n\n" + "Full travel detail. " * 1000
        self.assertEqual(strip_thinking("<think>private notes</think>\n" + answer), answer.strip())

    def test_unfinished_reasoning_is_never_displayed(self):
        self.assertEqual(strip_thinking("<think>unfinished private notes"), "")
        self.assertEqual(strip_thinking("Visible answer\n<think>unfinished"), "Visible answer")

    def test_encoded_and_multiple_blocks(self):
        self.assertEqual(
            strip_thinking("&lt;THINK&gt;one&lt;/THINK&gt;Hello<think>two</think> world"),
            "Hello world",
        )

    def test_output_limit_is_explicit_and_keeps_partial_answer(self):
        response = SimpleNamespace(content="## Day 1\nArrival", response_metadata={"finish_reason": "length"})
        text = visible_model_response(response)
        self.assertTrue(text.startswith(response.content))
        self.assertIn("incomplete", text)

    def test_reasoning_only_response_has_actionable_empty_state(self):
        response = SimpleNamespace(content="<think>unfinished", response_metadata={"finish_reason": "length"})
        self.assertIn("generate the plan again", visible_model_response(response))
        self.assertNotIn("unfinished", visible_model_response(response))

    def test_list_content_blocks_are_normalized_before_cleaning(self):
        self.assertEqual(
            strip_thinking([{"text": "<think>notes</think>## Plan"}, {"content": "Day 1"}]),
            "## Plan\nDay 1",
        )


class AgentResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from langgraph.checkpoint.memory import InMemorySaver

        saver = InMemorySaver()
        saver.setup = lambda: None
        # Import the real graph without opening a database or calling any provider.
        env = {
            "DATABASE_URL": "postgresql://user:password@localhost:5432/test",
            "GROQ_API_KEY": "test-groq-key",
            "TAVILY_API_KEY": "test-tavily-key",
        }
        with patch.dict(os.environ, env), patch("psycopg.connect"), patch(
            "langgraph.checkpoint.postgres.PostgresSaver", return_value=saver
        ):
            import backend
        cls.backend = backend

    def test_tools_retain_full_text_while_prompts_stay_bounded(self):
        from langchain_core.messages import AIMessage

        text = "Full source detail. " * 1000 + "END OF SOURCE"
        state = {"user_query": "A four day trip", "llm_calls": 0}
        with patch.object(self.backend, "search_flights", return_value=text), patch.object(
            self.backend, "tavily_mcp_search", new=AsyncMock(return_value=text)
        ):
            self.assertEqual(self.backend.flight_agent(state)["flight_results"], text)
            self.assertEqual(self.backend.hotel_agent(state)["hotel_results"], text)

        with patch.object(self.backend, "tavily_mcp_search", new=AsyncMock(return_value=[{"title": "Hotel", "content": "Full stay detail"}])):
            hotel_result = self.backend.hotel_agent(state)["hotel_results"]
        self.assertIn("Full stay detail", hotel_result)

        state.update(flight_results=text, hotel_results=text)
        with patch.object(self.backend, "llm") as llm:
            llm.invoke.return_value = AIMessage(content="<think>notes</think>## Itinerary\nAll days")
            result = self.backend.itinerary_agent(state)
            prompt = llm.invoke.call_args.args[0][1].content
            self.assertNotIn("END OF SOURCE", prompt)
            self.assertEqual(result["itinerary"], "## Itinerary\nAll days")
            self.assertNotIn("<think>", result["messages"][0].content)

    def test_final_response_is_cleaned_and_length_flagged(self):
        from langchain_core.messages import AIMessage

        state = dict(user_query="Dubai", flight_results="Flights", hotel_results="Hotels", itinerary="<think>old notes</think>Day 1")
        with patch.object(self.backend, "llm") as llm:
            llm.invoke.return_value = AIMessage(content="<think>new notes</think>## Plan", response_metadata={"finish_reason": "length"})
            result = self.backend.final_agent(state)
            self.assertNotIn("old notes", llm.invoke.call_args.args[0][1].content)
            self.assertNotIn("new notes", result["messages"][0].content)
            self.assertIn("incomplete", result["messages"][0].content)

    def test_hotel_snippets_are_not_cut_at_300_characters(self):
        from tools import tavily_tool

        content = "Hotel details. " * 100 + "END OF HOTEL"
        with patch.object(tavily_tool.client, "search", return_value={"results": [{"title": "Hotel", "url": "https://example.com", "content": content}]}):
            result = tavily_tool.tavily_search("Dubai hotels")
        self.assertIn(content, result)
        self.assertTrue(result.startswith("### 1. Hotel"))

    def test_flight_parser_accepts_list_input(self):
        from tools.flight_tool import parse_route

        self.assertEqual(parse_route(["from India", "to Dubai"]), ("DEL", "DXB"))


if __name__ == "__main__":
    unittest.main()
