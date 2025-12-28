import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch


import sys
import os
from unittest.mock import MagicMock


# Add src to path so we can import main (align with other tests)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Avoid importing heavy browser deps during unit tests
with patch.dict(sys.modules, {"camoufox.async_api": MagicMock(), "camoufox": MagicMock()}):
    import main


class TestRefreshRecaptchaToken(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # Ensure predictable globals between tests
        main.RECAPTCHA_TOKEN = None
        main.RECAPTCHA_EXPIRY = datetime.now(timezone.utc) - timedelta(days=365)

    async def test_force_refresh_always_fetches_new_token(self):
        main.RECAPTCHA_TOKEN = "cached"
        main.RECAPTCHA_EXPIRY = datetime.now(timezone.utc) + timedelta(seconds=60)

        with patch.object(main, "get_recaptcha_v3_token", new=AsyncMock(return_value="fresh")) as fetch:
            token = await main.refresh_recaptcha_token(force=True)

        self.assertEqual(token, "fresh")
        fetch.assert_awaited_once()

    async def test_does_not_refresh_when_token_is_still_valid(self):
        main.RECAPTCHA_TOKEN = "cached"
        main.RECAPTCHA_EXPIRY = datetime.now(timezone.utc) + timedelta(seconds=60)

        with patch.object(main, "get_recaptcha_v3_token", new=AsyncMock(return_value="fresh")) as fetch:
            token = await main.refresh_recaptcha_token(force=False)

        self.assertEqual(token, "cached")
        fetch.assert_not_awaited()

    async def test_refreshes_when_token_missing_or_expired(self):
        main.RECAPTCHA_TOKEN = None
        main.RECAPTCHA_EXPIRY = datetime.now(timezone.utc) - timedelta(seconds=1)

        with patch.object(main, "get_recaptcha_v3_token", new=AsyncMock(return_value="fresh")) as fetch:
            token = await main.refresh_recaptcha_token(force=False)

        self.assertEqual(token, "fresh")
        fetch.assert_awaited_once()

    async def test_failed_refresh_sets_short_backoff(self):
        now = datetime.now(timezone.utc)
        main.RECAPTCHA_TOKEN = None
        main.RECAPTCHA_EXPIRY = now - timedelta(seconds=1)

        with patch.object(main, "get_recaptcha_v3_token", new=AsyncMock(return_value=None)) as fetch:
            token = await main.refresh_recaptcha_token(force=False)

        self.assertIsNone(token)
        fetch.assert_awaited_once()
        self.assertGreater(main.RECAPTCHA_EXPIRY, now)
