import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from tests._stream_test_utils import BaseBridgeTest


class TestStreamUserscriptProxyLongDelayRespectsStreamDeadline(BaseBridgeTest):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        # This is the user-facing budget we should honor for proxy streaming.
        self.setup_config({"stream_total_timeout_seconds": 300})

    async def test_proxy_timeout_seconds_is_not_hardcoded_to_120(self) -> None:
        proxy_calls: dict[str, int] = {"count": 0}

        orig_proxy = self.main.fetch_lmarena_stream_via_userscript_proxy

        async def _proxy_stream(*args, **kwargs):  # noqa: ANN001
            proxy_calls["count"] += 1

            timeout_seconds = kwargs.get("timeout_seconds")
            self.assertGreaterEqual(
                int(timeout_seconds or 0),
                300,
                "Expected userscript-proxy stream timeout to respect stream_total_timeout_seconds",
            )

            resp = await orig_proxy(*args, **kwargs)
            self.assertIsNotNone(resp)

            job_id = str(resp.job_id)
            job = self.main._USERSCRIPT_PROXY_JOBS.get(job_id)
            self.assertIsInstance(job, dict)

            picked = job.get("picked_up_event")
            if isinstance(picked, asyncio.Event) and not picked.is_set():
                picked.set()
            job["picked_up_at_monotonic"] = float(self.main.time.monotonic())
            job["phase"] = "fetch"
            job["upstream_started_at_monotonic"] = float(self.main.time.monotonic())
            job["upstream_fetch_started_at_monotonic"] = float(self.main.time.monotonic())

            job["status_code"] = 200
            status_event = job.get("status_event")
            if isinstance(status_event, asyncio.Event):
                status_event.set()

            q = job.get("lines_queue")
            if isinstance(q, asyncio.Queue):
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
            patch.object(self.main, "refresh_recaptcha_token", AsyncMock(return_value="recaptcha-token")),
            patch.object(self.main, "fetch_lmarena_stream_via_userscript_proxy", proxy_mock),
            patch.object(self.main, "fetch_lmarena_stream_via_camoufox", AsyncMock(return_value=None)),
            patch.object(self.main, "fetch_lmarena_stream_via_chrome", AsyncMock(return_value=None)),
            patch("src.main.print"),
        ):
            # Mark proxy as active so strict-model routing prefers it.
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
