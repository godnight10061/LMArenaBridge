import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from tests._stream_test_utils import BaseBridgeTest


class TestStreamUserscriptProxyStatusZeroDoesNotRaise(BaseBridgeTest):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        # Ensure streaming takes the "no token configured" path (proxy-first).
        self.setup_config({"auth_tokens": []})

    async def test_stream_continues_when_proxy_status_unknown(self) -> None:
        proxy_calls: dict[str, int] = {"count": 0}

        orig_proxy = self.main.fetch_lmarena_stream_via_userscript_proxy

        async def _proxy_stream(*args, **kwargs):  # noqa: ANN001
            proxy_calls["count"] += 1
            resp = await orig_proxy(*args, **kwargs)
            self.assertIsNotNone(resp)

            job_id = str(resp.job_id)
            job = self.main._USERSCRIPT_PROXY_JOBS.get(job_id)
            self.assertIsInstance(job, dict)

            q = job.get("lines_queue")
            self.assertIsInstance(q, asyncio.Queue)

            # Intentionally do NOT set job["status_code"] or status_event. The bridge should not treat
            # status=0 as a fatal error; it should still parse the queued stream lines.
            await q.put('a0:"Hello"')
            await q.put('ad:{"finishReason":"stop"}')
            await q.put(None)

            job["done"] = True
            done_event = job.get("done_event")
            if isinstance(done_event, asyncio.Event):
                done_event.set()

            return resp

        proxy_mock = AsyncMock(side_effect=_proxy_stream)

        with (
            patch.object(self.main, "get_models") as get_models_mock,
            patch.object(self.main, "fetch_lmarena_stream_via_userscript_proxy", proxy_mock),
            patch("src.main.print"),
        ):
            self.main._touch_userscript_poll()
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
        self.assertGreaterEqual(proxy_calls["count"], 1)


if __name__ == "__main__":
    unittest.main()

