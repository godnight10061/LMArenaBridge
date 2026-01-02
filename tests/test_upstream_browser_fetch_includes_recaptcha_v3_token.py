import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx


class TestUpstreamBrowserFetchIncludesRecaptchaV3Token(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from src import main

        self.main = main
        self.main.dashboard_sessions.clear()
        self.main.chat_sessions.clear()
        self.main.api_key_usage.clear()

        self._temp_dir = tempfile.TemporaryDirectory()
        self._config_path = Path(self._temp_dir.name) / "config.json"
        self._config_path.write_text(
            json.dumps(
                {
                    "password": "admin",
                    "auth_tokens": ["auth-token-1"],
                    "upstream_via_browser": True,
                    "recaptcha_v2_enabled": False,
                    "api_keys": [{"name": "Test Key", "key": "test-key", "rpm": 999}],
                }
            ),
            encoding="utf-8",
        )

        self._orig_config_file = self.main.CONFIG_FILE
        self.main.CONFIG_FILE = str(self._config_path)

    async def asyncTearDown(self) -> None:
        self.main.CONFIG_FILE = self._orig_config_file
        self._temp_dir.cleanup()

    async def test_browser_fetch_includes_recaptcha_v3_outside_pytest(self) -> None:
        async def fake_chrome_fetch(*, http_method, url, payload, auth_token, timeout_seconds=120):
            # Browser fetch should mint a fresh token in-session (payload is cleared before fetch).
            self.assertEqual(payload.get("recaptchaV3Token"), "")
            return self.main.BrowserFetchStreamResponse(
                status_code=200,
                headers={},
                text='a0:"Hello"\nad:{"finishReason":"stop"}\n',
            )

        chrome_fetch_mock = AsyncMock(side_effect=fake_chrome_fetch)
        refresh_mock = AsyncMock(return_value="recaptcha-1")

        with patch.object(self.main, "get_models") as get_models_mock, patch.dict(
            os.environ,
            {"PYTEST_CURRENT_TEST": ""},
        ), patch.object(
            self.main,
            "find_chrome_executable",
            return_value="chrome.exe",
        ), patch.object(
            self.main,
            "fetch_lmarena_stream_via_chrome",
            chrome_fetch_mock,
        ), patch.object(
            self.main,
            "refresh_recaptcha_token",
            refresh_mock,
        ):
            get_models_mock.return_value = [
                {
                    "publicName": "test-model",
                    "id": "model-id",
                    "organization": "test-org",
                    "capabilities": {
                        "inputCapabilities": {"text": True},
                        "outputCapabilities": {"text": True},
                    },
                }
            ]

            transport = httpx.ASGITransport(app=self.main.app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                async with client.stream(
                    "POST",
                    "/api/v1/chat/completions",
                    headers={"Authorization": "Bearer test-key"},
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": True,
                    },
                ) as response:
                    body_text = (await response.aread()).decode("utf-8", errors="replace")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Hello", body_text)
        self.assertIn("[DONE]", body_text)
        chrome_fetch_mock.assert_awaited()
        refresh_mock.assert_not_awaited()
