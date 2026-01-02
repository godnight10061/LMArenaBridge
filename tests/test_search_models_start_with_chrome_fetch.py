import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx


class TestSearchModelsStartWithChromeFetch(unittest.IsolatedAsyncioTestCase):
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

    async def test_search_model_prefers_chrome_fetch_over_httpx(self) -> None:
        async def fake_chrome_fetch(*, http_method, url, payload, auth_token, timeout_seconds=120):
            self.assertEqual(payload.get("recaptchaV3Token"), "")
            return self.main.BrowserFetchStreamResponse(
                status_code=200,
                headers={},
                text='a0:"Hello"\nad:{"finishReason":"stop"}\n',
            )

        chrome_fetch_mock = AsyncMock(side_effect=fake_chrome_fetch)

        class FakeUpstreamAsyncClient:
            def __init__(self, *args, **kwargs):
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, *_args, **_kwargs):
                raise AssertionError("httpx upstream stream should not be used for search models")

        class HttpxShim:
            AsyncClient = FakeUpstreamAsyncClient

            def __getattr__(self, name: str):
                return getattr(httpx, name)

        with patch.object(self.main, "get_models") as get_models_mock, patch.object(
            self.main,
            "find_chrome_executable",
            return_value="chrome.exe",
        ), patch.object(
            self.main,
            "fetch_lmarena_stream_via_chrome",
            chrome_fetch_mock,
        ), patch.object(
            self.main,
            "get_async_camoufox",
            return_value=None,
        ), patch.object(
            self.main,
            "refresh_recaptcha_token",
            AsyncMock(return_value="recaptcha-1"),
        ), patch.object(
            self.main,
            "httpx",
            HttpxShim(),
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
        chrome_fetch_mock.assert_awaited()


if __name__ == "__main__":
    unittest.main()
