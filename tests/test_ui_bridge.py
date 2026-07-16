import unittest

from src import main


class _FakeAnchor:
    def __init__(self):
        self.clicked = False

    async def click(self, **_kwargs):
        self.clicked = True

    async def get_attribute(self, name):
        return "true" if name == "aria-checked" else None


class _FakeFrame:
    def __init__(self, anchor):
        self.anchor = anchor

    def locator(self, selector):
        assert selector == "#recaptcha-anchor"
        return self.anchor


class _FakeHandle:
    def __init__(self, frame):
        self.frame = frame

    async def content_frame(self):
        return self.frame


class _FakeIframe:
    def __init__(self, handle):
        self.first = self
        self.handle = handle

    async def wait_for(self, **_kwargs):
        return None

    async def element_handle(self):
        return self.handle


class _FakePage:
    def __init__(self):
        self.anchor = _FakeAnchor()
        frame = _FakeFrame(self.anchor)
        self.iframe = _FakeIframe(_FakeHandle(frame))

    def locator(self, selector):
        assert "size=normal" in selector
        return self.iframe

    async def wait_for_timeout(self, _timeout_ms):
        return None


class RecaptchaV2UiTests(unittest.IsolatedAsyncioTestCase):
    async def test_clicks_visible_recaptcha_v2_checkbox(self):
        page = _FakePage()

        handled = await main._click_visible_recaptcha_v2(page, timeout_ms=1)

        self.assertTrue(handled)
        self.assertTrue(page.anchor.clicked)


class CdpChromeLaunchTests(unittest.TestCase):
    def test_profile_path_with_spaces_stays_in_one_argument(self):
        profile = r"D:\bridge copy\chrome profile"

        command = main._build_cdp_chrome_command(
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            profile,
            9333,
        )

        self.assertIn(f"--user-data-dir={profile}", command)
        self.assertNotIn("--enable-automation", command)
        self.assertNotIn("--remote-debugging-pipe", command)


class _FakeTextLocator:
    def __init__(self, text="", visible=False):
        self.first = self
        self.text = text
        self.visible = visible

    async def count(self):
        return 1 if self.visible else 0

    async def is_visible(self):
        return self.visible

    async def inner_text(self):
        return self.text


class _FakeErrorPage:
    def get_by_text(self, value, exact=False):
        if isinstance(value, str) and value == "Something went wrong":
            return _FakeTextLocator(
                "Something went wrong with this response, please try again.",
                visible=True,
            )
        return _FakeTextLocator(
            "Trace ID: b197299198f45be16fb3608de4774e26",
            visible=True,
        )

    def get_by_role(self, role, name, exact=True):
        assert role == "button"
        assert name == "Copy trace ID"
        return _FakeTextLocator(visible=True)


class ArenaUiErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_detects_trace_id_error_variant(self):
        detail = await main._visible_arena_ui_error(_FakeErrorPage())

        self.assertEqual(
            detail,
            "Trace ID: b197299198f45be16fb3608de4774e26",
        )


if __name__ == "__main__":
    unittest.main()
