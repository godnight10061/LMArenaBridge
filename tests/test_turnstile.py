import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path so we can import main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Avoid importing heavy browser deps during unit tests
with patch.dict(sys.modules, {"camoufox.async_api": MagicMock(), "camoufox": MagicMock()}):
    import main


class _FakeElement:
    async def bounding_box(self):
        return {"x": 0, "y": 0, "width": 10, "height": 10}


class _FakeMouse:
    def __init__(self):
        self.click = AsyncMock(return_value=None)


class _FakePage:
    def __init__(self):
        self.mouse = _FakeMouse()

    async def query_selector(self, selector: str):
        return _FakeElement() if selector == "#cf-turnstile" else None


class TestTurnstile(unittest.IsolatedAsyncioTestCase):
    async def test_click_turnstile_clicks_widget(self):
        page = _FakePage()
        with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            clicked = await main.click_turnstile(page)
        self.assertTrue(clicked)
        page.mouse.click.assert_awaited()

