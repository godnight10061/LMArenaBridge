import os
import tempfile
import unittest
from pathlib import Path

from src.config import _apply_config_defaults
from src.recaptcha import find_chrome_executable


class ChromeOnlyConfigTests(unittest.TestCase):
    def test_legacy_browser_backend_settings_are_removed(self) -> None:
        config = {
            "camoufox_proxy_headless": True,
            "camoufox_fetch_window_mode": "hide",
            "chrome_fetch_window_mode": "hide",
        }

        _apply_config_defaults(config)

        self.assertFalse(any(key.startswith("camoufox_") for key in config))
        self.assertNotIn("chrome_fetch_window_mode", config)
        self.assertEqual(config["browser_window_mode"], "background")

    def test_legacy_profile_directory_remains_ignored(self) -> None:
        ignore_lines = {
            line.strip()
            for line in (Path(__file__).resolve().parents[1] / ".gitignore")
            .read_text(encoding="utf-8")
            .splitlines()
        }

        self.assertIn("camoufox_fetch/", ignore_lines)

    def test_browser_executable_honors_environment_override(self) -> None:
        previous = os.environ.get("LM_CHROME_EXECUTABLE")
        try:
            with tempfile.TemporaryDirectory(prefix="chrome-override-test-") as temp_dir:
                executable = Path(temp_dir) / "custom-chrome.exe"
                executable.write_bytes(b"fixture")
                os.environ["LM_CHROME_EXECUTABLE"] = str(executable)

                self.assertEqual(find_chrome_executable(), str(executable))
        finally:
            if previous is None:
                os.environ.pop("LM_CHROME_EXECUTABLE", None)
            else:
                os.environ["LM_CHROME_EXECUTABLE"] = previous

    def test_browser_executable_uses_config_then_playwright_fallback(self) -> None:
        environment_keys = (
            "LM_CHROME_EXECUTABLE",
            "CHROME_PATH",
            "PROGRAMFILES",
            "PROGRAMFILES(X86)",
            "LOCALAPPDATA",
            "PATH",
        )
        previous = {key: os.environ.get(key) for key in environment_keys}
        try:
            with tempfile.TemporaryDirectory(prefix="chrome-resolution-test-") as temp_dir:
                root = Path(temp_dir)
                configured = root / "configured-chrome.exe"
                playwright = root / "playwright-chromium.exe"
                configured.write_bytes(b"fixture")
                playwright.write_bytes(b"fixture")
                os.environ.pop("LM_CHROME_EXECUTABLE", None)
                os.environ.pop("CHROME_PATH", None)
                os.environ["PROGRAMFILES"] = str(root / "missing-program-files")
                os.environ["PROGRAMFILES(X86)"] = str(root / "missing-program-files-x86")
                os.environ["LOCALAPPDATA"] = str(root / "missing-local-app-data")
                os.environ["PATH"] = ""

                self.assertEqual(
                    find_chrome_executable(
                        {"chrome_fetch_executable": str(configured)},
                        playwright_executable=str(playwright),
                    ),
                    str(configured),
                )
                self.assertEqual(
                    find_chrome_executable({}, playwright_executable=str(playwright)),
                    str(playwright),
                )
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
