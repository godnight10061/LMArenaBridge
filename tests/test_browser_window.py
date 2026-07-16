import unittest

from src import main
from src.browser_window import chrome_window_args


class BrowserWindowTests(unittest.TestCase):
    def test_background_mode_is_headed_but_minimized_and_offscreen(self):
        args = chrome_window_args("background")

        self.assertIn("--start-minimized", args)
        self.assertIn("--window-position=-32000,-32000", args)
        self.assertFalse(any(arg.startswith("--headless") for arg in args))

    def test_visible_cdp_command_removes_background_placement(self):
        command = main._build_cdp_chrome_command(
            "chrome.exe", "profile", 9333, window_mode="visible"
        )

        self.assertIn("--window-position=0,0", command)
        self.assertNotIn("--start-minimized", command)
        self.assertNotIn("--window-position=-32000,-32000", command)


if __name__ == "__main__":
    unittest.main()
