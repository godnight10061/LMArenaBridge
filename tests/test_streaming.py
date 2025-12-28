import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path so we can import main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Avoid importing heavy browser deps during unit tests
with patch.dict(sys.modules, {"camoufox.async_api": MagicMock(), "camoufox": MagicMock()}):
    import main

from fastapi.testclient import TestClient
import httpx


class _FakeStreamResponse:
    def __init__(self, *, status_code: int, lines: list[str]):
        self.status_code = status_code
        self.headers = {}
        self._lines = lines

    def read(self) -> bytes:
        return "\n".join(self._lines).encode("utf-8")

    async def aread(self) -> bytes:
        return self.read()

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://lmarena.ai")
            response = httpx.Response(self.status_code, request=request, text=awaitable_text(self._lines))
            raise httpx.HTTPStatusError("error", request=request, response=response)

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamContext:
    def __init__(self, response: _FakeStreamResponse):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


def awaitable_text(lines: list[str]) -> str:
    return "\n".join(lines)


class TestStreamingChatCompletions(unittest.TestCase):
    def setUp(self):
        self._orig_models_json = None
        if os.path.exists("models.json"):
            with open("models.json", "r", encoding="utf-8") as f:
                self._orig_models_json = f.read()

        self.config = {
            "api_keys": [
                {"name": "Test Key", "key": "test-key", "rpm": 1000, "created": 1700000000}
            ],
            "auth_tokens": ["test-token"],
            "cf_clearance": "test-clearance",
            "password": "admin",
            "auth_token": "test-token",
            "usage_stats": {},
        }
        self.models = [
            {
                "id": "gpt-4",
                "publicName": "gpt-4",
                "organization": "openai",
                "capabilities": {"outputCapabilities": {"text": True}},
            },
            {
                "id": "img-model",
                "publicName": "img-model",
                "organization": "openai",
                "capabilities": {"outputCapabilities": {"image": True}},
            },
        ]

        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f)
        with open("models.json", "w", encoding="utf-8") as f:
            json.dump(self.models, f)

        main.chat_sessions.clear()
        main.api_key_usage.clear()

        # Speed up startup in TestClient
        self.initial_data_patcher = patch.object(main, "get_initial_data", new=AsyncMock(return_value=None))
        self.periodic_refresh_patcher = patch.object(main, "periodic_refresh_task", new=AsyncMock(return_value=None))
        self.recaptcha_patcher = patch.object(main, "refresh_recaptcha_token", new=AsyncMock(return_value="token"))
        self.initial_data_patcher.start()
        self.periodic_refresh_patcher.start()
        self.recaptcha_patcher.start()

    def tearDown(self):
        self.initial_data_patcher.stop()
        self.periodic_refresh_patcher.stop()
        self.recaptcha_patcher.stop()

        if os.path.exists("config.json"):
            os.remove("config.json")

        if self._orig_models_json is not None:
            with open("models.json", "w", encoding="utf-8") as f:
                f.write(self._orig_models_json)
        elif os.path.exists("models.json"):
            os.remove("models.json")

    def test_streaming_response_includes_done_and_content(self):
        citation_obj = {
            "toolCallId": "t1",
            "argsTextDelta": json.dumps({"source": {"title": "Example", "url": "https://example.com"}}),
        }
        lines = [
            'ag:"thinking..."',
            'a0:"Hello "',
            f"ac:{json.dumps(citation_obj)}",
            'a0:"world"',
            'ad:{"finishReason":"stop"}',
        ]
        fake_response = _FakeStreamResponse(status_code=200, lines=lines)

        def fake_stream(self, method, url, json=None, headers=None, timeout=None):
            return _FakeStreamContext(fake_response)

        with patch("httpx.AsyncClient.stream", new=fake_stream):
            with TestClient(main.app) as client:
                with client.stream(
                    "POST",
                    "/api/v1/chat/completions",
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                    headers={"Authorization": "Bearer test-key"},
                ) as resp:
                    body = "".join(resp.iter_text())

        self.assertEqual(resp.status_code, 200)
        self.assertIn("Hello", body)
        self.assertIn("[DONE]", body)

    def test_streaming_image_model_outputs_markdown(self):
        lines = [
            'a2:[{"type":"image","image":"https://img.example"}]',
            'ad:{"finishReason":"stop"}',
        ]
        fake_response = _FakeStreamResponse(status_code=200, lines=lines)

        def fake_stream(self, method, url, json=None, headers=None, timeout=None):
            return _FakeStreamContext(fake_response)

        with patch("httpx.AsyncClient.stream", new=fake_stream):
            with TestClient(main.app) as client:
                with client.stream(
                    "POST",
                    "/api/v1/chat/completions",
                    json={
                        "model": "img-model",
                        "messages": [{"role": "user", "content": "draw"}],
                        "stream": True,
                    },
                    headers={"Authorization": "Bearer test-key"},
                ) as resp:
                    body = "".join(resp.iter_text())

        self.assertEqual(resp.status_code, 200)
        self.assertIn("Generated Image", body)

    def test_streaming_rate_limit_yields_error_chunk(self):
        fake_response = _FakeStreamResponse(status_code=429, lines=[])

        def fake_stream(self, method, url, json=None, headers=None, timeout=None):
            return _FakeStreamContext(fake_response)

        with patch("httpx.AsyncClient.stream", new=fake_stream), patch(
            "asyncio.sleep", new=AsyncMock(return_value=None)
        ):
            with TestClient(main.app) as client:
                with client.stream(
                    "POST",
                    "/api/v1/chat/completions",
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                    headers={"Authorization": "Bearer test-key"},
                ) as resp:
                    body = "".join(resp.iter_text())

        self.assertEqual(resp.status_code, 200)
        self.assertIn("rate_limit_error", body)

    def test_streaming_recaptcha_400_retries_and_succeeds(self):
        # Ensure we have multiple auth tokens so retry path can rotate.
        self.config["auth_tokens"] = ["t1", "t2"]
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f)

        responses = [
            _FakeStreamResponse(status_code=400, lines=["recaptcha validation failed"]),
            _FakeStreamResponse(status_code=200, lines=['a0:"ok"', 'ad:{"finishReason":"stop"}']),
        ]

        def fake_stream(self, method, url, json=None, headers=None, timeout=None):
            return _FakeStreamContext(responses.pop(0))

        refresh_mock = AsyncMock(side_effect=["token1", "token2", "token3"])

        with patch("httpx.AsyncClient.stream", new=fake_stream), patch(
            "asyncio.sleep", new=AsyncMock(return_value=None)
        ), patch.object(main, "refresh_recaptcha_token", new=refresh_mock):
            with TestClient(main.app) as client:
                with client.stream(
                    "POST",
                    "/api/v1/chat/completions",
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                    headers={"Authorization": "Bearer test-key"},
                ) as resp:
                    body = "".join(resp.iter_text())

        self.assertEqual(resp.status_code, 200)
        self.assertIn("ok", body)
        self.assertGreaterEqual(refresh_mock.await_count, 2)
        forced_calls = [call for call in refresh_mock.await_args_list if call.kwargs.get("force") is True]
        self.assertGreaterEqual(len(forced_calls), 2)

    def test_streaming_401_yields_authentication_error_chunk(self):
        fake_response = _FakeStreamResponse(status_code=401, lines=[])

        def fake_stream(self, method, url, json=None, headers=None, timeout=None):
            return _FakeStreamContext(fake_response)

        with patch("httpx.AsyncClient.stream", new=fake_stream), patch(
            "asyncio.sleep", new=AsyncMock(return_value=None)
        ), patch.object(main, "remove_auth_token", new=MagicMock()):
            with TestClient(main.app) as client:
                with client.stream(
                    "POST",
                    "/api/v1/chat/completions",
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                    headers={"Authorization": "Bearer test-key"},
                ) as resp:
                    body = "".join(resp.iter_text())

        self.assertEqual(resp.status_code, 200)
        self.assertIn("authentication_error", body)

    def test_streaming_image_invalid_json_is_ignored(self):
        lines = [
            "a2:invalid-json",
            'ad:{"finishReason":"stop"}',
        ]
        fake_response = _FakeStreamResponse(status_code=200, lines=lines)

        def fake_stream(self, method, url, json=None, headers=None, timeout=None):
            return _FakeStreamContext(fake_response)

        with patch("httpx.AsyncClient.stream", new=fake_stream):
            with TestClient(main.app) as client:
                with client.stream(
                    "POST",
                    "/api/v1/chat/completions",
                    json={
                        "model": "img-model",
                        "messages": [{"role": "user", "content": "draw"}],
                        "stream": True,
                    },
                    headers={"Authorization": "Bearer test-key"},
                ) as resp:
                    body = "".join(resp.iter_text())

        self.assertEqual(resp.status_code, 200)
        self.assertIn("[DONE]", body)

    def test_streaming_existing_session_appends_history(self):
        lines1 = ['a0:"one"', 'ad:{"finishReason":"stop"}']
        lines2 = ['a0:"two"', 'ad:{"finishReason":"stop"}']
        responses = [_FakeStreamResponse(status_code=200, lines=lines1), _FakeStreamResponse(status_code=200, lines=lines2)]

        def fake_stream(self, method, url, json=None, headers=None, timeout=None):
            return _FakeStreamContext(responses.pop(0))

        with patch("httpx.AsyncClient.stream", new=fake_stream):
            with TestClient(main.app) as client:
                for content in ("hi", "hi"):
                    with client.stream(
                        "POST",
                        "/api/v1/chat/completions",
                        json={
                            "model": "gpt-4",
                            "messages": [{"role": "user", "content": content}],
                            "stream": True,
                        },
                        headers={"Authorization": "Bearer test-key"},
                    ) as resp:
                        _ = "".join(resp.iter_text())

        self.assertEqual(resp.status_code, 200)
        # Same first user message => same conversation id => second call appends history.
        self.assertTrue(main.chat_sessions["test-key"])
        session = next(iter(main.chat_sessions["test-key"].values()))
        self.assertGreaterEqual(len(session["messages"]), 4)
