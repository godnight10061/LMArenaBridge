import unittest
from unittest.mock import AsyncMock, patch


class TestRecaptchaChromeFallback(unittest.IsolatedAsyncioTestCase):
    async def test_get_recaptcha_token_uses_chrome_provider_when_available(self) -> None:
        from src import main

        main.DEBUG = False

        chrome_mock = AsyncMock(return_value="token-123")

        with patch.object(main, "get_config", return_value={}), patch.object(
            main, "get_recaptcha_v3_token_with_chrome", chrome_mock
        ), patch.object(main, "AsyncCamoufox") as camoufox_mock:
            token = await main.get_recaptcha_v3_token()

        self.assertEqual(token, "token-123")
        chrome_mock.assert_awaited()
        camoufox_mock.assert_not_called()

    async def test_get_recaptcha_v3_token_with_chrome_passes_explicit_headless_flag(self) -> None:
        from src import main

        main.DEBUG = False

        mock_page = AsyncMock()
        mock_page.title.return_value = "LMArena"
        mock_page.evaluate.side_effect = [
            "user-agent",
            "token-123",
        ]

        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page

        mock_playwright = AsyncMock()
        mock_playwright.chromium.launch_persistent_context.return_value = mock_context
        mock_playwright.__aenter__.return_value = mock_playwright

        window_mode_mock = AsyncMock()

        with (
            patch("playwright.async_api.async_playwright", return_value=mock_playwright),
            patch.object(main, "find_chrome_executable", return_value="C:/chrome.exe"),
            patch.object(main, "get_recaptcha_settings", return_value=("key", "action")),
            patch.object(main, "click_turnstile", AsyncMock(return_value=True)),
            patch.object(main.asyncio, "sleep", AsyncMock()),
            patch.object(main, "_get_arena_context_cookies", AsyncMock(return_value=[])),
            patch.object(main, "_upsert_browser_session_into_config", return_value=False),
            patch.object(main, "_maybe_apply_camoufox_window_mode", window_mode_mock),
        ):
            token = await main.get_recaptcha_v3_token_with_chrome({})

        self.assertEqual(token, "token-123")
        window_mode_mock.assert_awaited()
        self.assertFalse(window_mode_mock.await_args.kwargs.get("headless"))


if __name__ == "__main__":
    unittest.main()

