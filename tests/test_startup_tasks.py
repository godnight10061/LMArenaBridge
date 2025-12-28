import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path so we can import main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Avoid importing heavy browser deps during unit tests
with patch.dict(sys.modules, {"camoufox.async_api": MagicMock(), "camoufox": MagicMock()}):
    import main


class TestStartupTasks(unittest.IsolatedAsyncioTestCase):
    async def test_startup_event_runs_expected_steps(self):
        # Prevent noisy console output and avoid scheduling a never-ending background task.
        def _create_task(coro):
            coro.close()
            return MagicMock()

        cfg = {
            "password": "admin",
            "auth_token": "",
            "auth_tokens": [],
            "cf_clearance": "",
            "api_keys": [],
            "usage_stats": {},
        }

        with patch.object(main, "debug_print", new=MagicMock()), patch.object(
            main, "get_config", return_value=cfg
        ), patch.object(main, "save_config", new=MagicMock()) as save_config, patch.object(
            main, "get_models", return_value=[]
        ), patch.object(main, "save_models", new=MagicMock()) as save_models, patch.object(
            main, "load_usage_stats", new=MagicMock()
        ) as load_usage_stats, patch.object(
            main, "get_initial_data", new=AsyncMock(return_value=None)
        ) as get_initial_data, patch.object(
            main, "refresh_recaptcha_token", new=AsyncMock(return_value="token")
        ) as refresh_recaptcha_token, patch(
            "asyncio.create_task", new=_create_task
        ):
            await main.startup_event()

        save_config.assert_called()
        save_models.assert_called()
        load_usage_stats.assert_called_once()
        get_initial_data.assert_awaited()
        refresh_recaptcha_token.assert_awaited()

    async def test_periodic_refresh_task_continues_on_error(self):
        # First sleep returns immediately, initial refresh raises, then next sleep is cancelled to stop the loop.
        sleep_mock = AsyncMock(side_effect=[None, asyncio.CancelledError()])
        get_initial_data_mock = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(main, "debug_print", new=MagicMock()), patch(
            "asyncio.sleep", new=sleep_mock
        ), patch.object(main, "get_initial_data", new=get_initial_data_mock):
            with self.assertRaises(asyncio.CancelledError):
                await main.periodic_refresh_task()

        get_initial_data_mock.assert_awaited()
