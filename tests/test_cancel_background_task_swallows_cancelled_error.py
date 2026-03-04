import asyncio
import unittest

from src.browser_utils import _cancel_background_task


class TestCancelBackgroundTaskSwallowsCancelledError(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_background_task_does_not_raise(self) -> None:
        async def sleeper() -> None:
            await asyncio.sleep(60)

        task = asyncio.create_task(sleeper())
        await _cancel_background_task(task)
        await asyncio.sleep(0)
        self.assertTrue(task.done())


if __name__ == "__main__":
    unittest.main()

