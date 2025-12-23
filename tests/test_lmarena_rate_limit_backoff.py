import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx


class TestLMArenaRateLimitBackoff(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from src import main

        self.main = main
        self.main.dashboard_sessions.clear()
        self.main.chat_sessions.clear()

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

    async def test_stream_429_respects_retry_after_without_refreshing_recaptcha(self) -> None:
        real_async_client = httpx.AsyncClient
        upstream_calls = {"count": 0}

        def upstream_handler(request: httpx.Request) -> httpx.Response:
            upstream_calls["count"] += 1
            if upstream_calls["count"] == 1:
                return httpx.Response(
                    429,
                    headers={"Retry-After": "2"},
                    content=b"",
                )

            body = b'a0:"Hello"\nad:{"finishReason":"stop"}\n'
            return httpx.Response(200, content=body)

        upstream_transport = httpx.MockTransport(upstream_handler)

        def upstream_client_factory(*args, **kwargs) -> httpx.AsyncClient:
            return real_async_client(transport=upstream_transport)

        async def fake_refresh_recaptcha_token(*args, **kwargs) -> str:
            return "recaptcha-token"

        token_calls = {"count": 0}

        def fake_get_next_auth_token(exclude_tokens=None):
            token_calls["count"] += 1
            return "auth-token-1"

        sleep_mock = AsyncMock()

        with patch.object(self.main, "get_models") as get_models_mock, patch(
            "src.main.httpx.AsyncClient",
            side_effect=upstream_client_factory,
        ), patch.object(
            self.main,
            "refresh_recaptcha_token",
            side_effect=fake_refresh_recaptcha_token,
        ) as refresh_mock, patch.object(
            self.main,
            "get_next_auth_token",
            side_effect=fake_get_next_auth_token,
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

        self.assertEqual(upstream_calls["count"], 2)
        self.assertEqual(token_calls["count"], 1)
        self.assertEqual(refresh_mock.call_count, 2)

        sleep_mock.assert_any_await(2)
