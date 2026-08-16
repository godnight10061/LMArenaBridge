"""
Non-streaming 401 handling: refresh the LMArena auth session before giving up.

Reproduces issue #172: with a single (server-side invalidated) auth token, a
non-streaming POST to /api/v1/chat/completions used to fail with a misleading
"503: Max retries exceeded" because the pre-stream request path only rotated
tokens on 401 and never tried to refresh the session (unlike the mid-stream
path).

These tests drive the real endpoint (ASGI) and the real `make_request_with_retry`
code path, mocking only the network layer (cloudscraper) and the refresh calls.
"""

import base64
import json
import time
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import requests

from tests._stream_test_utils import BaseBridgeTest


def _build_arena_auth_token(expires_in: int = 3600) -> str:
    """Build a plausible, non-expired `arena-auth-prod-v1` base64 session token."""
    session = {
        "access_token": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0LXVzZXIifQ.fakesig",
        "refresh_token": "refresh-chain-123",
        "token_type": "bearer",
        "expires_in": expires_in,
        "expires_at": int(time.time()) + expires_in,
    }
    raw = base64.b64encode(json.dumps(session).encode("utf-8")).decode("ascii").rstrip("=")
    return "base64-" + raw


def _make_response(status_code: int, text: str = "") -> requests.Response:
    """Build a `requests.Response` as returned by cloudscraper."""
    resp = requests.Response()
    resp.status_code = int(status_code)
    resp._content = text.encode("utf-8")
    resp.headers = {}
    resp.url = "https://arena.ai/nextjs-api/stream/create-evaluation"
    return resp


class _ScriptedScraper:
    """cloudscraper scraper stand-in: pops the next scripted response per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: ANN001
        self.calls.append((url, json, headers))
        return self.responses.pop(0)

    def put(self, url, json=None, headers=None, timeout=None):  # noqa: ANN001
        return self.post(url, json=json, headers=headers, timeout=timeout)


MODEL = {
    "publicName": "test-search-model",
    "id": "model-id",
    "organization": "test-org",
    "capabilities": {
        "inputCapabilities": {"text": True},
        "outputCapabilities": {"search": True},
    },
}

SUCCESS_BODY = 'a0:"Hello from LMArena"\nad:{"finishReason":"stop"}\n'


class Test401TokenRefreshNonStream(BaseBridgeTest):
    """Non-streaming create-evaluation path (`make_request_with_retry`)."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        # Ensure a stale ephemeral token from a previous test cannot leak into this one.
        self.main.EPHEMERAL_ARENA_AUTH_TOKEN = None

    def _patch_refresh(self, *, http=None, pool=None, supabase=None):
        return (
            patch.object(
                self.main,
                "refresh_arena_auth_token_via_lmarena_http",
                AsyncMock(return_value=http),
            ),
            patch.object(
                self.main,
                "maybe_refresh_expired_auth_tokens",
                AsyncMock(return_value=pool),
            ),
            patch.object(
                self.main,
                "refresh_arena_auth_token_via_supabase",
                AsyncMock(return_value=supabase),
            ),
        )

    async def _post_chat_completion(self):
        transport = httpx.ASGITransport(app=self.main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/v1/chat/completions",
                headers={"Authorization": "Bearer test-key"},
                json={
                    "model": "test-search-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                timeout=30.0,
            )

    async def test_401_with_single_token_refreshes_session_and_succeeds(self) -> None:
        """A 401 on the only configured token must refresh the session and complete the request.

        Before the fix this failed with "503: Max retries exceeded" (the issue #172 symptom).
        """
        old_token = _build_arena_auth_token()
        refreshed_token = _build_arena_auth_token(expires_in=3500)
        self.setup_config({"auth_tokens": [old_token]})
        scraper = _ScriptedScraper(
            [
                _make_response(401, '{"error":"Unauthorized"}'),
                _make_response(200, SUCCESS_BODY),
            ]
        )

        refresh_patches = self._patch_refresh(http=refreshed_token)
        with patch.object(self.main, "get_models", return_value=[MODEL]), patch.object(
            self.main, "refresh_recaptcha_token", AsyncMock(return_value="recaptcha-token")
        ), patch("cloudscraper.create_scraper", return_value=scraper), patch("src.main.asyncio.sleep", AsyncMock()), refresh_patches[0], refresh_patches[1], refresh_patches[2]:
            response = await self._post_chat_completion()

        self.assertEqual(response.status_code, 200, msg=response.text)
        self.assertIn("Hello from LMArena", response.text)
        # The refreshed session must be remembered for subsequent requests.
        self.assertEqual(self.main.EPHEMERAL_ARENA_AUTH_TOKEN, refreshed_token)
        # The retry must have used the refreshed token in the cookie header.
        self.assertIn(refreshed_token, scraper.calls[-1][2].get("Cookie", ""))

    async def test_401_refresh_failure_returns_401_not_503(self) -> None:
        """When refresh fails and no other token exists, report a truthful auth error instead of
        the misleading "503: Max retries exceeded" from the issue."""
        old_token = _build_arena_auth_token()
        self.setup_config({"auth_tokens": [old_token]})
        scraper = _ScriptedScraper([_make_response(401, '{"error":"Unauthorized"}')] * 3)

        refresh_patches = self._patch_refresh()
        with patch.object(self.main, "get_models", return_value=[MODEL]), patch.object(
            self.main, "refresh_recaptcha_token", AsyncMock(return_value="recaptcha-token")
        ), patch("cloudscraper.create_scraper", return_value=scraper), patch("src.main.asyncio.sleep", AsyncMock()), refresh_patches[0], refresh_patches[1], refresh_patches[2]:
            response = await self._post_chat_completion()

        error_body = response.json()["error"]
        self.assertEqual(error_body["code"], 401)
        self.assertEqual(error_body["type"], "authentication_error")
        self.assertIn("auth token", error_body["message"].lower())
        self.assertNotIn("503", error_body["message"])
        self.assertNotIn("Max retries exceeded", error_body["message"])

    async def test_200_response_parses_content(self) -> None:
        """A plain 200 from the cloudscraper transport must produce a chat completion.

        Regression: the non-streaming path called `await response.aread()` on a sync
        `requests.Response`, which crashed with AttributeError even on success.
        """
        self.setup_config({"auth_tokens": [_build_arena_auth_token()]})
        scraper = _ScriptedScraper([_make_response(200, SUCCESS_BODY)])

        with patch.object(self.main, "get_models", return_value=[MODEL]), patch.object(
            self.main, "refresh_recaptcha_token", AsyncMock(return_value="recaptcha-token")
        ), patch("cloudscraper.create_scraper", return_value=scraper), patch("src.main.asyncio.sleep", AsyncMock()):
            response = await self._post_chat_completion()

        self.assertEqual(response.status_code, 200, msg=response.text)
        self.assertIn("Hello from LMArena", response.text)

    async def test_401_falls_back_to_next_configured_token_when_refresh_fails(self) -> None:
        """Refresh failure must still rotate to the next configured token (regression)."""
        first = _build_arena_auth_token()
        second = _build_arena_auth_token(expires_in=3400)
        self.setup_config({"auth_tokens": [first, second]})
        scraper = _ScriptedScraper(
            [
                _make_response(401, '{"error":"Unauthorized"}'),
                _make_response(200, SUCCESS_BODY),
            ]
        )

        refresh_patches = self._patch_refresh()
        with patch.object(self.main, "get_models", return_value=[MODEL]), patch.object(
            self.main, "refresh_recaptcha_token", AsyncMock(return_value="recaptcha-token")
        ), patch("cloudscraper.create_scraper", return_value=scraper), patch("src.main.asyncio.sleep", AsyncMock()), refresh_patches[0], refresh_patches[1], refresh_patches[2]:
            response = await self._post_chat_completion()

        self.assertEqual(response.status_code, 200, msg=response.text)
        self.assertIn("Hello from LMArena", response.text)


if __name__ == "__main__":
    unittest.main()

