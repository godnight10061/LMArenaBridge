import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

class TestRecaptchaTokenPool(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from src import main

        self.main = main
        self.main.chat_sessions.clear()
        self.main.api_key_usage.clear()

        self._temp_dir = tempfile.TemporaryDirectory()
        self._config_path = Path(self._temp_dir.name) / "config.json"
        self._config_path.write_text(
            json.dumps(
                {
                    "password": "admin",
                    "auth_tokens": ["auth-token-1"],
                    "recaptcha_token_pool_size": 3,
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

    async def test_refresh_recaptcha_token_prefetches_and_consumes_pool(self) -> None:
        calls = {"count": 0}

        async def fake_get_tokens(*, auth_token: str | None, count: int):
            calls["count"] += 1
            self.assertEqual(auth_token, "auth-token-1")
            self.assertEqual(count, 3)
            return ["token-1", "token-2", "token-3"]

        with patch.object(self.main, "get_recaptcha_v3_tokens", side_effect=fake_get_tokens):
            first = await self.main.refresh_recaptcha_token(auth_token="auth-token-1", force_new=True)
            second = await self.main.refresh_recaptcha_token(auth_token="auth-token-1", force_new=False)

        self.assertEqual(first, "token-3")
        self.assertEqual(second, "token-2")
        self.assertEqual(calls["count"], 1)
