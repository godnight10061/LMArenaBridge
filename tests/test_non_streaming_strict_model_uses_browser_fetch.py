import unittest
from unittest.mock import AsyncMock, patch

import httpx
import cloudscraper

from tests._stream_test_utils import BaseBridgeTest


class TestNonStreamingStrictModelUsesBrowserFetch(BaseBridgeTest):
    async def test_gemini_grounding_non_stream_uses_chrome_fetch(self) -> None:
        chrome_resp = self.main.BrowserFetchStreamResponse(
            status_code=200,
            headers={},
            text='a0:"Hello"\nad:{"finishReason":"stop"}\n',
            method="POST",
            url="https://lmarena.ai/nextjs-api/stream/create-evaluation",
        )
        chrome_fetch_mock = AsyncMock(return_value=chrome_resp)

        def fail_if_cloudscraper_called(*args, **kwargs):  # noqa: ARG001
            raise AssertionError("cloudscraper should not be used for strict models")

        with patch.object(self.main, "get_models") as get_models_mock, patch.object(
            self.main,
            "fetch_lmarena_stream_via_chrome",
            chrome_fetch_mock,
        ), patch.object(
            cloudscraper,
            "create_scraper",
            new=fail_if_cloudscraper_called,
        ), patch(
            "src.main.print",
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
                        "stream": False,
                    },
                    timeout=30.0,
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["choices"][0]["message"]["content"], "Hello")
        chrome_fetch_mock.assert_awaited()


if __name__ == "__main__":
    unittest.main()
