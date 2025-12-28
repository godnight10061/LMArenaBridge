import json
import os
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path so we can import main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Avoid importing heavy browser deps during unit tests
with patch.dict(sys.modules, {"camoufox.async_api": MagicMock(), "camoufox": MagicMock()}):
    import main


class _FakeContext:
    async def cookies(self):
        return [{"name": "cf_clearance", "value": "cf-clearance-test"}]


class _FakePage:
    def __init__(self, body: str):
        self._body = body
        self.context = _FakeContext()

    async def route(self, *args, **kwargs):
        return None

    async def goto(self, *args, **kwargs):
        return None

    async def wait_for_function(self, *args, **kwargs):
        return True

    async def content(self):
        return self._body


class _FakeBrowser:
    def __init__(self, page: _FakePage):
        self._page = page

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def new_page(self):
        return self._page


class _FakeRouteResponse:
    def __init__(self, body_bytes: bytes):
        self._body_bytes = body_bytes

    async def body(self):
        return self._body_bytes


class _FakeRoute:
    def __init__(self, url: str, body_bytes: bytes):
        self.request = MagicMock()
        self.request.url = url
        self._resp = _FakeRouteResponse(body_bytes)
        self.fulfill = AsyncMock(return_value=None)
        self.continue_ = AsyncMock(return_value=None)

    async def fetch(self):
        return self._resp


class _FakePageWithRoute(_FakePage):
    def __init__(self, body: str, js_text: str):
        super().__init__(body)
        self._route_handler = None
        self._js_text = js_text

    async def route(self, pattern, handler):
        self._route_handler = handler

    async def goto(self, *args, **kwargs):
        # Simulate a single JS chunk fetch that contains the Next-Action IDs.
        if self._route_handler:
            route = _FakeRoute(
                "https://lmarena.ai/_next/static/chunks/abcd.js",
                self._js_text.encode("utf-8"),
            )
            await self._route_handler(route)
        return None


class _FakeMouse:
    async def move(self, *args, **kwargs):
        return None

    async def wheel(self, *args, **kwargs):
        return None


class _FakeRecaptchaPage:
    def __init__(self):
        self.mouse = _FakeMouse()
        self._poll_count = 0

    async def goto(self, *args, **kwargs):
        return None

    async def title(self):
        return "LM Arena"

    async def wait_for_load_state(self, *args, **kwargs):
        return None

    async def evaluate(self, script: str):
        # Library readiness checks
        if "window.grecaptcha" in script:
            return True
        # Init global result
        if "__token_result = 'PENDING'" in script:
            self._poll_count = 0
            return None
        # Poll result
        if script.strip() == "mw:window.__token_result":
            self._poll_count += 1
            return "tok-test" if self._poll_count >= 1 else "PENDING"
        return None


class _FakeRecaptchaPageTurnstile(_FakeRecaptchaPage):
    def __init__(self):
        super().__init__()
        self._title_calls = 0
        self._lib_calls = 0

    async def title(self):
        self._title_calls += 1
        return "Just a moment..." if self._title_calls == 1 else "LM Arena"

    async def evaluate(self, script: str):
        if "window.grecaptcha" in script:
            self._lib_calls += 1
            return self._lib_calls >= 2  # fail first, succeed second
        return await super().evaluate(script)


class _FakeRecaptchaPageNoLibrary(_FakeRecaptchaPage):
    async def evaluate(self, script: str):
        if "window.grecaptcha" in script:
            return False
        return await super().evaluate(script)


class _FakeRecaptchaPageJsError(_FakeRecaptchaPage):
    async def evaluate(self, script: str):
        if script.strip() == "mw:window.__token_result":
            return "ERROR: boom"
        return await super().evaluate(script)


class _FakeRecaptchaContext:
    async def add_cookies(self, *args, **kwargs):
        return None

    async def new_page(self):
        return _FakeRecaptchaPage()


class _FakeRecaptchaContextWithPage(_FakeRecaptchaContext):
    def __init__(self, page):
        self._page = page

    async def new_page(self):
        return self._page


class _FakeRecaptchaBrowser:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def new_context(self):
        return _FakeRecaptchaContext()


class _FakeRecaptchaBrowserWithPage(_FakeRecaptchaBrowser):
    def __init__(self, page):
        self._page = page

    async def new_context(self):
        return _FakeRecaptchaContextWithPage(self._page)


class TestBackgroundTasks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._orig_models_json = None
        if os.path.exists("models.json"):
            with open("models.json", "r", encoding="utf-8") as f:
                self._orig_models_json = f.read()

    async def asyncTearDown(self):
        if os.path.exists("config.json"):
            os.remove("config.json")

        if self._orig_models_json is not None:
            with open("models.json", "w", encoding="utf-8") as f:
                f.write(self._orig_models_json)
        elif os.path.exists("models.json"):
            os.remove("models.json")

    async def test_get_initial_data_saves_cf_clearance_and_models(self):
        body = (
            '{\\"initialModels\\":[{'
            '\\"id\\":\\"gpt-4\\",\\"publicName\\":\\"gpt-4\\",\\"organization\\":\\"openai\\",'
            '\\"capabilities\\":{\\"outputCapabilities\\":{\\"text\\":true}}'
            '}],\\"initialModelAId\\":\\"gpt-4\\"}'
        )

        def fake_camoufox(*args, **kwargs):
            return _FakeBrowser(_FakePage(body))

        with patch.object(main, "AsyncCamoufox", new=fake_camoufox), patch(
            "asyncio.sleep", new=AsyncMock(return_value=None)
        ):
            await main.get_initial_data()

        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(cfg.get("cf_clearance"), "cf-clearance-test")

        with open("models.json", "r", encoding="utf-8") as f:
            models = json.load(f)
        self.assertEqual(models[0]["publicName"], "gpt-4")

    async def test_get_recaptcha_v3_token_captures_token(self):
        # Ensure config has a clearance cookie (optional path)
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump({"cf_clearance": "cf-clearance-test"}, f)

        def fake_camoufox(*args, **kwargs):
            return _FakeRecaptchaBrowser()

        with patch.object(main, "AsyncCamoufox", new=fake_camoufox), patch.object(
            main, "click_turnstile", new=AsyncMock(return_value=False)
        ), patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            token = await main.get_recaptcha_v3_token()

        self.assertEqual(token, "tok-test")
        self.assertEqual(main.RECAPTCHA_TOKEN, "tok-test")
        self.assertGreater(main.RECAPTCHA_EXPIRY, datetime.now(timezone.utc))

    async def test_get_recaptcha_handles_turnstile_and_library_retry(self):
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump({"cf_clearance": "cf-clearance-test"}, f)

        def fake_camoufox(*args, **kwargs):
            return _FakeRecaptchaBrowserWithPage(_FakeRecaptchaPageTurnstile())

        with patch.object(main, "AsyncCamoufox", new=fake_camoufox), patch.object(
            main, "click_turnstile", new=AsyncMock(return_value=True)
        ), patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            token = await main.get_recaptcha_v3_token()

        self.assertEqual(token, "tok-test")

    async def test_get_recaptcha_returns_none_when_library_never_loads(self):
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump({"cf_clearance": "cf-clearance-test"}, f)

        def fake_camoufox(*args, **kwargs):
            return _FakeRecaptchaBrowserWithPage(_FakeRecaptchaPageNoLibrary())

        with patch.object(main, "AsyncCamoufox", new=fake_camoufox), patch.object(
            main, "click_turnstile", new=AsyncMock(return_value=False)
        ), patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            token = await main.get_recaptcha_v3_token()

        self.assertIsNone(token)

    async def test_get_recaptcha_returns_none_on_js_error(self):
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump({"cf_clearance": "cf-clearance-test"}, f)

        def fake_camoufox(*args, **kwargs):
            return _FakeRecaptchaBrowserWithPage(_FakeRecaptchaPageJsError())

        with patch.object(main, "AsyncCamoufox", new=fake_camoufox), patch.object(
            main, "click_turnstile", new=AsyncMock(return_value=False)
        ), patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            token = await main.get_recaptcha_v3_token()

        self.assertIsNone(token)

    async def test_get_initial_data_extracts_next_action_ids(self):
        body = (
            '{\\"initialModels\\":[{'
            '\\"id\\":\\"gpt-4\\",\\"publicName\\":\\"gpt-4\\",\\"organization\\":\\"openai\\",'
            '\\"capabilities\\":{\\"outputCapabilities\\":{\\"text\\":true}}'
            '}],\\"initialModelAId\\":\\"gpt-4\\"}'
        )
        js_text = (
            '(0,a.createServerReference)("upload123",b.callServer,void 0,c.findSourceMapURL,"generateUploadUrl");'
            '(0,a.createServerReference)("signed123",b.callServer,void 0,c.findSourceMapURL,"getSignedUrl");'
        )

        def fake_camoufox(*args, **kwargs):
            return _FakeBrowser(_FakePageWithRoute(body, js_text))

        with patch.object(main, "AsyncCamoufox", new=fake_camoufox), patch(
            "asyncio.sleep", new=AsyncMock(return_value=None)
        ):
            await main.get_initial_data()

        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(cfg.get("next_action_upload"), "upload123")
        self.assertEqual(cfg.get("next_action_signed_url"), "signed123")
