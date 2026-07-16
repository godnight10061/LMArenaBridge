import unittest

from src.main import _system_prompt_log_metadata


class LoggingSafetyTests(unittest.TestCase):
    def test_system_prompt_log_contains_metadata_not_content(self) -> None:
        marker = "PRIVATE_SYSTEM_MARKER"

        message = _system_prompt_log_metadata([{"role": "system", "content": marker}], marker)

        self.assertEqual(message, "System prompt messages=1 chars=21")
        self.assertNotIn(marker, message)


if __name__ == "__main__":
    unittest.main()
