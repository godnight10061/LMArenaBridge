import json
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from src import main


class ChallengePreflightTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._original_config_file = main.CONFIG_FILE
        self._original_models_file = main.MODELS_FILE
        self._original_debug = main.DEBUG
        self._runtime = tempfile.TemporaryDirectory(prefix="challenge-preflight-")
        runtime_path = Path(self._runtime.name)
        now = int(time.time())
        config_path = runtime_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "api_keys": [],
                    "auth_token": "",
                    "auth_tokens": [],
                    "managed_account": {},
                    "managed_account_history": [],
                    "browser_ui_all_text_models": True,
                    "challenge_recovery": {
                        "schema": 1,
                        "phase": "cooldown",
                        "model": "gemini-3.5-flash",
                        "first_challenge_at": now,
                        "retry_not_before": now + 900,
                        "replacement_used": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        models_path = runtime_path / "models.json"
        models_path.write_text(
            json.dumps(
                [
                    {
                        "id": "gemini-3.5-flash",
                        "publicName": "gemini-3.5-flash",
                        "organization": "google",
                        "capabilities": {
                            "inputCapabilities": {"text": True},
                            "outputCapabilities": {"text": True},
                        },
                    }
                ]
            ),
            encoding="utf-8",
        )
        main.CONFIG_FILE = str(config_path)
        main.MODELS_FILE = str(models_path)
        main.DEBUG = False
        main.api_key_usage.clear()

    async def asyncTearDown(self) -> None:
        main.CONFIG_FILE = self._original_config_file
        main.MODELS_FILE = self._original_models_file
        main.DEBUG = self._original_debug
        main.api_key_usage.clear()
        self._runtime.cleanup()

    async def test_cooldown_returns_immediate_classified_503(self) -> None:
        transport = httpx.ASGITransport(app=main.app, raise_app_exceptions=False)
        started = time.monotonic()
        async with httpx.AsyncClient(
            transport=transport, base_url="http://bridge.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-3.5-flash",
                    "messages": [{"role": "user", "content": "fixture prompt"}],
                    "stream": False,
                },
            )
        elapsed = time.monotonic() - started

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.headers.get("x-lmbridge-error-code"), "challenge_unresolved"
        )
        self.assertEqual(response.headers.get("x-lmbridge-challenge-phase"), "cooldown")
        self.assertIn(int(response.headers.get("retry-after", "0")), {899, 900})
        self.assertLess(elapsed, 3)


if __name__ == "__main__":
    unittest.main()
