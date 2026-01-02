import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

class TestRecaptchaPoolSizeDefault(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from src import main

        self.main = main
        self.main.chat_sessions.clear()
        self.main.api_key_usage.clear()

        self._temp_dir = tempfile.TemporaryDirectory()
        self._config_path = Path(self._temp_dir.name) / "config.json"
        # config.json will NOT set recaptcha_token_pool_size,
        # so it should default to 3 from the main.py setdefault.
        self._config_path.write_text(
            json.dumps(
                {
                    "password": "admin",
                    "auth_tokens": ["auth-token-1"],
                }
            ),
            encoding="utf-8",
        )
        self._orig_config_file = self.main.CONFIG_FILE
        self.main.CONFIG_FILE = str(self._config_path)

        if hasattr(self.main, "RECAPTCHA_TOKEN_POOLS"):
            self.main.RECAPTCHA_TOKEN_POOLS.clear()
        if hasattr(self.main, "RECAPTCHA_POOL_EXPIRIES"):
            self.main.RECAPTCHA_POOL_EXPIRIES.clear()

    async def asyncTearDown(self) -> None:
        self.main.CONFIG_FILE = self._orig_config_file
        self._temp_dir.cleanup()

    async def test_recaptcha_token_pool_size_defaults_to_3(self) -> None:
        calls = {"count": 0}

        async def fake_get_tokens(*, auth_token: str | None, count: int):
            calls["count"] += 1
            self.assertEqual(auth_token, "auth-token-1")
            self.assertEqual(count, 3) # Assert that the default pool size is 3
            return [f"token-{i}" for i in range(count)]

        with patch.object(self.main, "get_recaptcha_v3_tokens", side_effect=fake_get_tokens):
            await self.main.refresh_recaptcha_token(auth_token="auth-token-1", force_new=True)

        self.assertEqual(calls["count"], 1)

