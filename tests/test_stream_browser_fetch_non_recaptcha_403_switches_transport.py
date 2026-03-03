import unittest
from unittest.mock import AsyncMock, patch

import httpx

from tests._stream_test_utils import BaseBridgeTest


class TestStreamBrowserFetchNonRecaptcha403SwitchesTransport(BaseBridgeTest):
    async def test_chrome_403_access_denied_switches_to_camoufox(self) -> None:
        chrome_calls: dict[str, int] = {"count": 0}
        camoufox_calls: dict[str, int] = {"count": 0}

        chrome_resp = self.main.BrowserFetchStreamResponse(
            status_code=403,
            headers={},
            text='{"error":"Access denied"}',
            method="POST",
            url="https://lmarena.ai/nextjs-api/stream/create-evaluation",
        )

        async def _chrome_stream(*args, **kwargs):  # noqa: ANN001
            chrome_calls["count"] += 1
            return chrome_resp

        chrome_mock = AsyncMock(side_effect=_chrome_stream)

        camoufox_resp = self.main.BrowserFetchStreamResponse(
            status_code=200,
            headers={},
            text='a0:"Hello"\nad:{"finishReason":"stop"}\n',
            method="POST",
            url="https://lmarena.ai/nextjs-api/stream/create-evaluation",
        )

        async def _camoufox_stream(*args, **kwargs):  # noqa: ANN001
            camoufox_calls["count"] += 1
            return camoufox_resp

        camoufox_mock = AsyncMock(side_effect=_camoufox_stream)

        with (
            patch.object(self.main, "get_models") as get_models_mock,
            patch.object(self.main, "refresh_recaptcha_token", AsyncMock(return_value="recaptcha-token")),
            patch.object(self.main, "fetch_lmarena_stream_via_chrome", chrome_mock),
            patch.object(self.main, "fetch_lmarena_stream_via_camoufox", camoufox_mock),
            patch("src.main.print"),
        ):
            get_models_mock.return_value = [
                {
                    "publicName": "gemini-3-pro-grounding",
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
                        "model": "gemini-3-pro-grounding",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": True,
                    },
                    timeout=30.0,
                )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Hello", response.text)
        self.assertIn("[DONE]", response.text)
        self.assertGreaterEqual(chrome_calls["count"], 1)
        self.assertGreaterEqual(camoufox_calls["count"], 1)


if __name__ == "__main__":
    unittest.main()

