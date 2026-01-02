import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx


class TestStream429DoesNotRefreshRecaptchaOnChrome(unittest.IsolatedAsyncioTestCase):
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

    async def test_429_on_chrome_fetch_does_not_refresh_recaptcha(self) -> None:
        call_state: dict[str, int] = {"chrome_fetch": 0}

        async def fake_chrome_fetch(*, http_method, url, payload, auth_token, timeout_seconds=120, **_kwargs):
            call_state["chrome_fetch"] += 1
            if call_state["chrome_fetch"] == 1:
                return self.main.BrowserFetchStreamResponse(
                    status_code=429,
                    headers={},
                    text=json.dumps({"error": "rate_limited"}),
                )
            return self.main.BrowserFetchStreamResponse(
                status_code=200,
                headers={},
                text='a0:"Hello"\nad:{"finishReason":"stop"}\n',
            )

        chrome_fetch_mock = AsyncMock(side_effect=fake_chrome_fetch)
        refresh_mock = AsyncMock(return_value="recaptcha-1")
        sleep_mock = AsyncMock()

        with patch.object(self.main, "get_models") as get_models_mock, patch.object(
            self.main,
            "find_chrome_executable",
            return_value="chrome.exe",
        ), patch.object(
            self.main,
            "get_async_camoufox",
            return_value=None,
        ), patch.object(
            self.main,
            "fetch_lmarena_stream_via_chrome",
            chrome_fetch_mock,
        ), patch.object(
            self.main,
            "refresh_recaptcha_token",
            refresh_mock,
        ), patch(
            "src.main.asyncio.sleep",
            sleep_mock,
        ):
            get_models_mock.return_value = [
                {
                    "publicName": "test-search-model",
                    "id": "model-id",
                    "organization": "test-org",
                    "capabilities": {
                        "inputCapabilities": {"text": True},
                        "outputCapabilities": {"search": True},
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
                        "model": "test-search-model",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": True,
                    },
                ) as response:
                    body_text = (await response.aread()).decode("utf-8", errors="replace")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Hello", body_text)
        self.assertIn("[DONE]", body_text)
        self.assertEqual(call_state["chrome_fetch"], 2)
        refresh_mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()

