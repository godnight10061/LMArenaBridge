import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from tests._stream_test_utils import BaseBridgeTest


class TestStreamUserscriptProxyNonRecaptcha403Fallback(BaseBridgeTest):
    async def test_stream_proxy_non_recaptcha_403_falls_back_to_chrome_fetch(self) -> None:
        proxy_calls: dict[str, int] = {"count": 0}
        chrome_calls: dict[str, int] = {"count": 0}

        orig_proxy = self.main.fetch_lmarena_stream_via_userscript_proxy

        async def _proxy_stream(*args, **kwargs):  # noqa: ANN001
            proxy_calls["count"] += 1
            resp = await orig_proxy(*args, **kwargs)
            self.assertIsNotNone(resp)

            job_id = str(resp.job_id)
            job = self.main._USERSCRIPT_PROXY_JOBS.get(job_id)
            self.assertIsInstance(job, dict)

            picked = job.get("picked_up_event")
            if isinstance(picked, asyncio.Event) and not picked.is_set():
                picked.set()

            job["phase"] = "fetch"
            job["picked_up_at_monotonic"] = float(self.main.time.monotonic())
            job["upstream_started_at_monotonic"] = float(self.main.time.monotonic())
            job["upstream_fetch_started_at_monotonic"] = float(self.main.time.monotonic())

            # Important: keep the initial status at 200 so the stream handler doesn't take the early 403 path.
            # We'll flip it to 403 after __aenter__ returns to simulate "late" proxy status updates.
            job["status_code"] = 200
            job["headers"] = {}

            q = job.get("lines_queue")
            self.assertIsInstance(q, asyncio.Queue)
            await q.put('{"error":"Access denied"}')
            await q.put(None)

            job["done"] = True
            done_event = job.get("done_event")
            if isinstance(done_event, asyncio.Event) and not done_event.is_set():
                done_event.set()

            async def _flip_status() -> None:
                await asyncio.sleep(0)
                job["status_code"] = 403
                status_event = job.get("status_event")
                if isinstance(status_event, asyncio.Event) and not status_event.is_set():
                    status_event.set()

            asyncio.create_task(_flip_status())
            return resp

        proxy_mock = AsyncMock(side_effect=_proxy_stream)

        chrome_resp = self.main.BrowserFetchStreamResponse(
            status_code=200,
            headers={},
            text='a0:"Hello"\nad:{"finishReason":"stop"}\n',
            method="POST",
            url="https://lmarena.ai/nextjs-api/stream/create-evaluation",
        )

        async def _chrome_stream(*args, **kwargs):  # noqa: ANN001
            chrome_calls["count"] += 1
            # Proxy is a backup transport; force browser transports to fail on the first attempt
            # so the userscript-proxy path is exercised.
            if chrome_calls["count"] == 1:
                return None
            return chrome_resp

        chrome_mock = AsyncMock(side_effect=_chrome_stream)

        with (
            patch.object(self.main, "get_models") as get_models_mock,
            patch.object(self.main, "refresh_recaptcha_token", AsyncMock(return_value="recaptcha-token")),
            patch.object(self.main, "fetch_lmarena_stream_via_userscript_proxy", proxy_mock),
            patch.object(self.main, "fetch_lmarena_stream_via_chrome", chrome_mock),
            patch.object(self.main, "fetch_lmarena_stream_via_camoufox", AsyncMock(return_value=None)),
            patch("src.main.print"),
        ):
            # Mark proxy as active so strict-model routing prefers it initially.
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
        self.assertGreaterEqual(chrome_calls["count"], 1)


if __name__ == "__main__":
    unittest.main()

