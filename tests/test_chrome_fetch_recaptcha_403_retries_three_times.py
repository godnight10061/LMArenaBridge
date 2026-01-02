import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


class TestChromeFetchRecaptcha403RetriesThreeTimes(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from src import main

        self.main = main
        self.main.dashboard_sessions.clear()
        self.main.chat_sessions.clear()
        self.main.api_key_usage.clear()

        self._temp_dir = tempfile.TemporaryDirectory()
        self._profile_dir = Path(self._temp_dir.name) / "chrome_grecaptcha"

    async def asyncTearDown(self) -> None:
        self._temp_dir.cleanup()

    async def test_chrome_fetch_retries_multiple_times_when_recaptcha_fails(self) -> None:
        class FakeKeyboard:
            async def type(self, _text: str) -> None:
                return None

            async def press(self, _key: str) -> None:
                return None

        class FakeLocator:
            async def wait_for(self, **_kwargs) -> None:
                return None

            async def click(self, **_kwargs) -> None:
                return None

        call_state: dict[str, int] = {"token_exec": 0, "fetch": 0}

        class FakeContext:
            async def clear_cookies(self) -> None:
                return None

            async def add_cookies(self, _cookies: list[dict]) -> None:
                return None

            async def new_page(self):
                return FakePage()

            async def cookies(self) -> list[dict]:
                return []

            async def close(self) -> None:
                return None

        class FakePage:
            def __init__(self) -> None:
                self.context = FakeContext()
                self.keyboard = FakeKeyboard()

            async def goto(self, *_args, **_kwargs) -> None:
                return None

            async def wait_for_function(self, *_args, **_kwargs) -> None:
                return None

            async def evaluate(self, script: str, arg=None):
                script = (script or "").strip()
                if script == "() => navigator.userAgent":
                    return "Mozilla/5.0 TestUA"
                if "g.execute(sitekey" in script:
                    call_state["token_exec"] += 1
                    return f"recaptcha-token-{call_state['token_exec']}"
                if "const res = await fetch" in script:
                    call_state["fetch"] += 1
                    if call_state["fetch"] <= 2:
                        return {
                            "status": 403,
                            "headers": {},
                            "text": json.dumps({"error": "recaptcha validation failed"}),
                        }
                    return {
                        "status": 200,
                        "headers": {},
                        "text": 'a0:"Hello"\nad:{"finishReason":"stop"}\n',
                    }
                return None

        class FakeChromium:
            async def launch_persistent_context(self, **_kwargs) -> FakeContext:
                return FakeContext()

        class FakePlaywright:
            def __init__(self) -> None:
                self.chromium = FakeChromium()

        class FakeAsyncPlaywrightCM:
            async def __aenter__(self) -> FakePlaywright:
                return FakePlaywright()

            async def __aexit__(self, exc_type, exc, tb) -> bool:
                return False

        def fake_async_playwright():
            return FakeAsyncPlaywrightCM()

        async def fake_get_chat_textarea_locator(*_args, **_kwargs) -> FakeLocator:
            return FakeLocator()

        payload = {"recaptchaV3Token": ""}

        with patch.object(self.main, "async_playwright", fake_async_playwright), patch.object(
            self.main,
            "find_chrome_executable",
            return_value="chrome.exe",
        ), patch.object(
            self.main,
            "get_browser_profile_dir",
            return_value=self._profile_dir,
        ), patch.object(
            self.main,
            "get_chromium_user_agent_for_playwright",
            AsyncMock(return_value="Mozilla/5.0 TestUA"),
        ), patch.object(
            self.main,
            "wait_for_cloudflare_challenge_to_clear",
            AsyncMock(return_value=None),
        ), patch.object(
            self.main,
            "get_chat_textarea_locator",
            AsyncMock(side_effect=fake_get_chat_textarea_locator),
        ), patch.object(
            self.main,
            "save_config",
            lambda *_args, **_kwargs: None,
        ), patch.object(
            self.main,
            "get_config",
            return_value={
                "cf_clearance": "cf",
                "cf_bm": "bm",
                "browser_cookies": {"other": "x"},
                "recaptcha_headless": True,
            },
        ):
            response = await self.main.fetch_lmarena_stream_via_chrome(
                http_method="POST",
                url="https://lmarena.ai/nextjs-api/stream/create-evaluation",
                payload=payload,
                auth_token="auth-token-1",
                timeout_seconds=5,
            )

        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(call_state["fetch"], 3)
        self.assertGreaterEqual(call_state["token_exec"], 3)


if __name__ == "__main__":
    unittest.main()

