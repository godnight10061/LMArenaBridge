import unittest

from src.live_gemini_check import (
    _bounded_request_delay,
    _select_model_candidates,
    _successful_candidate_models,
)


class LiveModelSelectionTests(unittest.TestCase):
    def test_request_delay_is_bounded(self):
        self.assertEqual(_bounded_request_delay(-1), 0.0)
        self.assertEqual(_bounded_request_delay(5), 5.0)
        self.assertEqual(_bounded_request_delay(90), 30.0)

    def test_selects_distinct_provider_diverse_models_from_live_records(self):
        records = [
            {"id": "gemini-3.5-flash", "owned_by": "google"},
            {"id": "gpt-5.2-chat-latest", "owned_by": "openai"},
            {"id": "claude-sonnet-4-6", "owned_by": "anthropic"},
            {"id": "grok-4.3", "owned_by": "xai"},
            {"id": "qwen3.7-plus", "owned_by": "alibaba"},
            {"id": "deepseek-v4-flash", "owned_by": "deepseek"},
            {"id": "gpt-5.4-search", "owned_by": "openai"},
        ]

        selected = _select_model_candidates(records, primary_model="gemini-3.5-flash", limit=5)

        self.assertEqual(len(selected), 5)
        self.assertEqual(len({item["model"] for item in selected}), 5)
        self.assertNotIn("gpt-5.4-search", {item["model"] for item in selected})
        self.assertGreaterEqual(len({item["provider"] for item in selected}), 5)

    def test_candidate_success_requirement_excludes_mandatory_gemini(self):
        attempts = [
            {"model": "gemini-3.5-flash", "status": "passed", "mandatory": True},
            {"model": "gpt-5.2-chat-latest", "status": "passed"},
            {"model": "claude-sonnet-4-6", "status": "passed"},
            {"model": "grok-4.3", "status": "failed"},
        ]

        successful = _successful_candidate_models(attempts, primary_model="gemini-3.5-flash")

        self.assertEqual(successful, {"gpt-5.2-chat-latest", "claude-sonnet-4-6"})


if __name__ == "__main__":
    unittest.main()
