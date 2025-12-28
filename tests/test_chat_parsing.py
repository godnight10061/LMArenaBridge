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
import hashlib


class TestChatParsing(unittest.TestCase):
    def setUp(self):
        self._orig_models_json = None
        if os.path.exists("models.json"):
            with open("models.json", "r", encoding="utf-8") as f:
                self._orig_models_json = f.read()

        self.config = {
            "api_keys": [{"name": "Test Key", "key": "test-key", "rpm": 1000, "created": 1700000000}],
            "auth_tokens": ["t1", "t2", "t3"],
            "cf_clearance": "test-clearance",
            "password": "admin",
            "auth_token": "t1",
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

        # Speed up startup
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

    def test_non_stream_parses_reasoning_and_citations(self):
        citation_obj = {
            "toolCallId": "t1",
            "argsTextDelta": json.dumps({"source": {"title": "Example", "url": "https://example.com"}}),
        }
        upstream_text = "\n".join(
            [
                'ag:"thinking..."',
                'a0:"Hello "',
                f"ac:{json.dumps(citation_obj)}",
                'a0:"world"',
                'ad:{"finishReason":"stop"}',
            ]
        )

        response_success = MagicMock()
        response_success.status_code = 200
        response_success.text = upstream_text
        response_success.headers = {}
        response_success.raise_for_status.return_value = None

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=response_success)):
            with TestClient(main.app) as client:
                resp = client.post(
                    "/api/v1/chat/completions",
                    json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer test-key"},
                )

        self.assertEqual(resp.status_code, 200)
        message = resp.json()["choices"][0]["message"]
        self.assertIn("Hello world", message["content"])
        self.assertIn("Sources:", message["content"])
        self.assertEqual(message["reasoning_content"], "thinking...")

    def test_rate_limit_error_returns_openai_error(self):
        from httpx import HTTPStatusError, Request, Response

        request = Request("POST", "https://lmarena.ai")
        response_obj = Response(429, request=request, text="too many requests")
        error = HTTPStatusError("429 Too Many Requests", request=request, response=response_obj)

        response_rate_limited = MagicMock()
        response_rate_limited.status_code = 429
        response_rate_limited.text = "too many requests"
        response_rate_limited.headers = {}
        response_rate_limited.raise_for_status.side_effect = error

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=response_rate_limited)), patch(
            "asyncio.sleep", new=AsyncMock(return_value=None)
        ):
            with TestClient(main.app) as client:
                resp = client.post(
                    "/api/v1/chat/completions",
                    json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer test-key"},
                )

        self.assertEqual(resp.status_code, 200)
        err = resp.json()["error"]
        self.assertEqual(err["type"], "rate_limit_error")

    def test_timeout_returns_openai_error(self):
        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=httpx.TimeoutException("timeout"))):
            with TestClient(main.app) as client:
                resp = client.post(
                    "/api/v1/chat/completions",
                    json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer test-key"},
                )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["error"]["type"], "timeout_error")

    def test_unexpected_error_returns_openai_error(self):
        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=RuntimeError("boom"))):
            with TestClient(main.app) as client:
                resp = client.post(
                    "/api/v1/chat/completions",
                    json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer test-key"},
                )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["error"]["type"], "internal_error")

    def test_empty_upstream_response_returns_error(self):
        response_empty = MagicMock()
        response_empty.status_code = 200
        response_empty.text = 'a3:"upstream error"'
        response_empty.headers = {}
        response_empty.raise_for_status.return_value = None

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=response_empty)):
            with TestClient(main.app) as client:
                resp = client.post(
                    "/api/v1/chat/completions",
                    json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer test-key"},
                )

        self.assertEqual(resp.status_code, 200)
        err = resp.json()["error"]
        self.assertEqual(err["type"], "upstream_error")

    def test_bad_request_uses_upstream_error_message(self):
        from httpx import HTTPStatusError, Request, Response

        request = Request("POST", "https://lmarena.ai")
        response_obj = Response(400, request=request, json={"error": "bad thing"})
        error = HTTPStatusError("400 Bad Request", request=request, response=response_obj)

        response_bad = MagicMock()
        response_bad.status_code = 400
        response_bad.text = '{"error":"bad thing"}'
        response_bad.headers = {}
        response_bad.raise_for_status.side_effect = error

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=response_bad)):
            with TestClient(main.app) as client:
                resp = client.post(
                    "/api/v1/chat/completions",
                    json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer test-key"},
                )

        self.assertEqual(resp.status_code, 200)
        self.assertIn("Bad Request", resp.json()["error"]["message"])

    def test_unauthorized_returns_authentication_error(self):
        from httpx import HTTPStatusError, Request, Response

        request = Request("POST", "https://lmarena.ai")
        response_obj = Response(401, request=request, text="unauthorized")
        error = HTTPStatusError("401 Unauthorized", request=request, response=response_obj)

        response_unauth = MagicMock()
        response_unauth.status_code = 401
        response_unauth.text = "unauthorized"
        response_unauth.headers = {}
        response_unauth.raise_for_status.side_effect = error

        with patch.object(main, "remove_auth_token", new=MagicMock()), patch(
            "httpx.AsyncClient.post", new=AsyncMock(return_value=response_unauth)
        ), patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            with TestClient(main.app) as client:
                resp = client.post(
                    "/api/v1/chat/completions",
                    json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer test-key"},
                )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["error"]["type"], "authentication_error")

    def test_forbidden_returns_forbidden_error(self):
        from httpx import HTTPStatusError, Request, Response

        request = Request("POST", "https://lmarena.ai")
        response_obj = Response(403, request=request, text="forbidden")
        error = HTTPStatusError("403 Forbidden", request=request, response=response_obj)

        response_forbidden = MagicMock()
        response_forbidden.status_code = 403
        response_forbidden.text = "forbidden"
        response_forbidden.headers = {}
        response_forbidden.raise_for_status.side_effect = error

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=response_forbidden)):
            with TestClient(main.app) as client:
                resp = client.post(
                    "/api/v1/chat/completions",
                    json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer test-key"},
                )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["error"]["type"], "forbidden_error")

    def test_not_found_returns_not_found_error(self):
        from httpx import HTTPStatusError, Request, Response

        request = Request("POST", "https://lmarena.ai")
        response_obj = Response(404, request=request, text="not found")
        error = HTTPStatusError("404 Not Found", request=request, response=response_obj)

        response_not_found = MagicMock()
        response_not_found.status_code = 404
        response_not_found.text = "not found"
        response_not_found.headers = {}
        response_not_found.raise_for_status.side_effect = error

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=response_not_found)):
            with TestClient(main.app) as client:
                resp = client.post(
                    "/api/v1/chat/completions",
                    json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer test-key"},
                )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["error"]["type"], "not_found_error")

    def test_server_error_returns_server_error(self):
        from httpx import HTTPStatusError, Request, Response

        request = Request("POST", "https://lmarena.ai")
        response_obj = Response(500, request=request, text="server error")
        error = HTTPStatusError("500 Server Error", request=request, response=response_obj)

        response_server_error = MagicMock()
        response_server_error.status_code = 500
        response_server_error.text = "server error"
        response_server_error.headers = {}
        response_server_error.raise_for_status.side_effect = error

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=response_server_error)):
            with TestClient(main.app) as client:
                resp = client.post(
                    "/api/v1/chat/completions",
                    json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer test-key"},
                )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["error"]["type"], "server_error")

    def test_bad_request_without_error_field_uses_default_message(self):
        from httpx import HTTPStatusError, Request, Response

        request = Request("POST", "https://lmarena.ai")
        response_obj = Response(400, request=request, json={"foo": "bar"})
        error = HTTPStatusError("400 Bad Request", request=request, response=response_obj)

        response_bad = MagicMock()
        response_bad.status_code = 400
        response_bad.text = '{"foo":"bar"}'
        response_bad.headers = {}
        response_bad.raise_for_status.side_effect = error

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=response_bad)):
            with TestClient(main.app) as client:
                resp = client.post(
                    "/api/v1/chat/completions",
                    json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer test-key"},
                )

        self.assertEqual(resp.status_code, 200)
        self.assertIn("Invalid request parameters", resp.json()["error"]["message"])
        self.assertEqual(resp.json()["error"]["type"], "bad_request_error")

    def test_unknown_status_includes_response_body_when_json_parseable(self):
        from httpx import HTTPStatusError, Request, Response

        request = Request("POST", "https://lmarena.ai")
        response_obj = Response(418, request=request, json={"detail": "teapot"})
        error = HTTPStatusError("418 I'm a teapot", request=request, response=response_obj)

        response_teapot = MagicMock()
        response_teapot.status_code = 418
        response_teapot.text = '{"detail":"teapot"}'
        response_teapot.headers = {}
        response_teapot.raise_for_status.side_effect = error

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=response_teapot)):
            with TestClient(main.app) as client:
                resp = client.post(
                    "/api/v1/chat/completions",
                    json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer test-key"},
                )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["error"]["type"], "upstream_error")
        self.assertIn("teapot", resp.json()["error"]["message"])

    def test_image_generation_non_stream_sets_markdown_content(self):
        response_image = MagicMock()
        response_image.status_code = 200
        response_image.text = (
            'a2:[{"type":"image","image":"https://img.example"}]\n'
            'ad:{"finishReason":"stop"}'
        )
        response_image.headers = {}
        response_image.raise_for_status.return_value = None

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *args, **kwargs):
                return response_image

            async def put(self, *args, **kwargs):
                return response_image

        with patch.object(main.httpx, "AsyncClient", return_value=_Client()):
            with TestClient(main.app) as client:
                resp = client.post(
                    "/api/v1/chat/completions",
                    json={"model": "img-model", "messages": [{"role": "user", "content": "draw"}]},
                    headers={"Authorization": "Bearer test-key"},
                )

        self.assertEqual(resp.status_code, 200)
        content = resp.json()["choices"][0]["message"]["content"]
        self.assertIn("![Generated Image]", content)

    def test_existing_session_followup_uses_same_conversation(self):
        # Two requests with the same first user message should map to the same conversation_id.
        resp1 = MagicMock(status_code=200, text='a0:"Hi"', headers={})
        resp1.raise_for_status.return_value = None
        resp2 = MagicMock(status_code=200, text='a0:"Again"', headers={})
        resp2.raise_for_status.return_value = None

        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=[resp1, resp2])):
            with TestClient(main.app) as client:
                r1 = client.post(
                    "/api/v1/chat/completions",
                    json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer test-key"},
                )
                r2 = client.post(
                    "/api/v1/chat/completions",
                    json={
                        "model": "gpt-4",
                        "messages": [
                            {"role": "user", "content": "hi"},
                            {"role": "user", "content": "follow up"},
                        ],
                    },
                    headers={"Authorization": "Bearer test-key"},
                )

        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        self.assertIn("Again", r2.json()["choices"][0]["message"]["content"])

    def test_retry_endpoint_uses_put(self):
        # Seed a session where the last stored message is a user message (no assistant yet),
        # so the API treats this as a retry.
        api_key_str = "test-key"
        model_public_name = "gpt-4"
        first_user_message = "hi"
        conversation_key = f"{api_key_str}_{model_public_name}_{first_user_message[:100]}"
        conversation_id = hashlib.sha256(conversation_key.encode()).hexdigest()[:16]

        main.chat_sessions[api_key_str][conversation_id] = {
            "conversation_id": "sess123",
            "model": model_public_name,
            "messages": [
                {"id": "assistant-1", "role": "assistant", "content": "prev"},
                {"id": "user-2", "role": "user", "content": "hi"},
            ],
        }

        response_put = MagicMock(status_code=200, text='a0:"Retried"', headers={})
        response_put.raise_for_status.return_value = None

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def put(self, *args, **kwargs):
                return response_put

        with patch.object(main.httpx, "AsyncClient", return_value=_Client()):
            with TestClient(main.app) as client:
                r = client.post(
                    "/api/v1/chat/completions",
                    json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer test-key"},
                )

        self.assertEqual(r.status_code, 200)
        self.assertIn("Retried", r.json()["choices"][0]["message"]["content"])
