import hashlib
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from tests._stream_test_utils import BaseBridgeTest, FakeStreamContext, FakeStreamResponse


class TestStreamNoDeltaFollowupRegeneratesIds(BaseBridgeTest):
    async def test_followup_retry_regenerates_ids_after_no_delta(self) -> None:
        payloads: list[dict] = []
        urls: list[str] = []

        api_key_str = "test-key"
        model_public_name = "test-search-model"
        first_user_message = "Hello"
        conversation_key = f"{api_key_str}_{model_public_name}_{first_user_message[:100]}"
        conversation_id = hashlib.sha256(conversation_key.encode()).hexdigest()[:16]

        self.main.chat_sessions.setdefault(api_key_str, {})[conversation_id] = {
            "conversation_id": "arena-session-1",
            "model": model_public_name,
            "messages": [
                {"id": "user-initial", "role": "user", "content": "Hello"},
                {"id": "assistant-initial", "role": "assistant", "content": "Hi there"},
            ],
        }

        def fake_stream(self, method, url, json=None, headers=None, timeout=None):  # noqa: ARG001
            payload = dict(json or {})
            payloads.append(payload)
            urls.append(str(url))

            if len(payloads) == 1:
                return FakeStreamContext(
                    FakeStreamResponse(
                        status_code=200,
                        headers={},
                        text='a3:"temporary upstream issue"\n',
                    )
                )

            first = payloads[0]
            same_ids = (
                payload.get("userMessageId") == first.get("userMessageId")
                and payload.get("modelAMessageId") == first.get("modelAMessageId")
                and payload.get("modelBMessageId") == first.get("modelBMessageId")
            )
            if same_ids:
                return FakeStreamContext(
                    FakeStreamResponse(
                        status_code=400,
                        headers={},
                        text='{"error":"duplicate ids"}',
                    )
                )

            return FakeStreamContext(
                FakeStreamResponse(
                    status_code=200,
                    headers={},
                    text='a0:"Hello follow-up"\nad:{"finishReason":"stop"}\n',
                )
            )

        with patch.object(self.main, "get_models") as get_models_mock, patch.object(
            self.main,
            "refresh_recaptcha_token",
            AsyncMock(return_value="recaptcha-token"),
        ), patch.object(
            httpx.AsyncClient,
            "stream",
            new=fake_stream,
        ), patch(
            "src.main.print",
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
                response = await client.post(
                    "/api/v1/chat/completions",
                    headers={"Authorization": "Bearer test-key"},
                    json={
                        "model": "test-search-model",
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi there"},
                            {"role": "user", "content": "Follow-up question"},
                        ],
                        "stream": True,
                    },
                    timeout=30.0,
                )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Hello follow-up", response.text)
        self.assertIn("[DONE]", response.text)
        self.assertGreaterEqual(len(payloads), 2)
        self.assertTrue(urls and "/post-to-evaluation/" in urls[0])
        self.assertEqual(payloads[0].get("id"), payloads[1].get("id"))
        for key in ("userMessageId", "modelAMessageId", "modelBMessageId"):
            with self.subTest(key=key):
                self.assertNotEqual(payloads[0].get(key), payloads[1].get(key))


if __name__ == "__main__":
    unittest.main()
