import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx


class TestIssue27RecaptchaFallback(unittest.IsolatedAsyncioTestCase):
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
                    "api_keys": [
                        {
                            "name": "Test Key",
                            "key": "test-key",
                            "rpm": 999,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        self._orig_config_file = self.main.CONFIG_FILE
        self.main.CONFIG_FILE = str(self._config_path)

    async def asyncTearDown(self) -> None:
        self.main.CONFIG_FILE = self._orig_config_file
        self._temp_dir.cleanup()

    async def test_403_recaptcha_switches_to_browser_fetch(self) -> None:
        real_async_client = httpx.AsyncClient

        def upstream_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"error": "recaptcha validation failed"})

        upstream_transport = httpx.MockTransport(upstream_handler)

        def upstream_client_factory(*args, **kwargs) -> httpx.AsyncClient:
            return real_async_client(transport=upstream_transport)

        refresh_mock = AsyncMock(side_effect=["recaptcha-1", "recaptcha-2"])
        sleep_mock = AsyncMock()
        fetch_mock = AsyncMock(
            return_value=self.main.BrowserFetchStreamResponse(
                status_code=200,
                headers={},
                text='a0:"Hello"\nad:{"finishReason":"stop"}\n',
            )
        )

        with patch.object(self.main, "get_models") as get_models_mock, patch(
            "src.main.httpx.AsyncClient",
            side_effect=upstream_client_factory,
        ), patch.object(
            self.main,
            "refresh_recaptcha_token",
            refresh_mock,
        ), patch.object(
            self.main,
            "find_chrome_executable",
            return_value="chrome.exe",
        ), patch.object(
            self.main,
            "fetch_lmarena_stream_via_chrome",
            fetch_mock,
        ), patch(
            "src.main.asyncio.sleep",
            sleep_mock,
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
            async with real_async_client(transport=transport, base_url="http://test") as client:
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
        self.assertGreaterEqual(refresh_mock.call_count, 2)
        fetch_mock.assert_awaited()
