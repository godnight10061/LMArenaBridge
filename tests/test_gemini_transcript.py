import unittest

from src.gemini_transcript import build_transcript


class GeminiTranscriptTests(unittest.TestCase):
    def test_single_user_turn_keeps_existing_prompt_shape(self):
        result = build_transcript([{"role": "user", "content": "hello"}])

        self.assertEqual(result.prompt, "hello")
        self.assertFalse(result.context_truncated)
        self.assertEqual(result.included_role_counts, {"user": 1})

    def test_preserves_ordered_roles_and_tool_results_but_excludes_reasoning(self):
        result = build_transcript(
            [
                {"role": "system", "content": "Follow the policy."},
                {"role": "user", "content": "Find the value."},
                {
                    "role": "assistant",
                    "content": "I will call the lookup tool.",
                    "reasoning_content": "private chain of thought",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"key":"nonce"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "NONCE_VALUE",
                },
                {"role": "user", "content": "What was the value?"},
            ]
        )

        self.assertLess(result.prompt.index("[SYSTEM]"), result.prompt.index("[USER]"))
        self.assertEqual(result.prompt.count("[SYSTEM]"), 1)
        self.assertLess(result.prompt.index("[TOOL_CALL id=call-1"), result.prompt.index("NONCE_VALUE"))
        self.assertLess(result.prompt.index("NONCE_VALUE"), result.prompt.rindex("What was the value?"))
        self.assertNotIn("private chain of thought", result.prompt)
        self.assertEqual(result.role_counts["assistant"], 1)
        self.assertEqual(result.role_counts["tool"], 1)

    def test_truncation_removes_oldest_whole_tool_unit(self):
        messages = [
            {"role": "system", "content": "Always answer with the requested nonce."},
            {"role": "user", "content": "old question"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "old-call",
                        "function": {"name": "old_tool", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "old-call", "content": "OLD_RESULT"},
            {"role": "user", "content": "new question " + ("x" * 120)},
        ]
        result = build_transcript(messages, max_characters=180)

        self.assertTrue(result.context_truncated)
        self.assertNotIn("OLD_RESULT", result.prompt)
        self.assertNotIn("old-call", result.prompt)
        self.assertIn("new question", result.prompt)
        self.assertIn("Always answer", result.prompt)
        self.assertGreater(result.omitted_message_count, 0)
        self.assertLessEqual(result.rendered_characters, 180)

    def test_preserves_instruction_position_in_source_order(self):
        result = build_transcript(
            [
                {"role": "user", "content": "first"},
                {"role": "developer", "content": "middle"},
                {"role": "user", "content": "last"},
            ]
        )

        self.assertLess(result.prompt.index("first"), result.prompt.index("middle"))
        self.assertLess(result.prompt.index("middle"), result.prompt.index("last"))


if __name__ == "__main__":
    unittest.main()
