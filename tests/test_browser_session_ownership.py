import asyncio
import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import account_recovery


def _authenticated_cookie() -> str:
    payload = {
        "expires_at": int(time.time()) + 3600,
        "user": {
            "aud": "authenticated",
            "role": "authenticated",
            "is_anonymous": False,
        },
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode(
        "ascii"
    )
    return "base64-" + encoded.rstrip("=")


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_for_cdp(port: int, timeout: float = 20.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    url = f"http://127.0.0.1:{port}/json/version"
    while asyncio.get_running_loop().time() < deadline:
        try:
            response = await asyncio.to_thread(urllib.request.urlopen, url, timeout=1)
            response.close()
            return
        except (OSError, urllib.error.URLError):
            await asyncio.sleep(0.2)
    raise TimeoutError("Chrome CDP endpoint did not become ready")


@unittest.skipUnless(os.name == "nt", "Windows browser ownership contract")
class TestRecoveryBrowserOwnership(unittest.IsolatedAsyncioTestCase):
    async def test_borrowed_cdp_context_preserves_browser_and_reads_auth_cookie(self):
        with tempfile.TemporaryDirectory(prefix="lmarena-cdp-test-") as temp_dir:
            profile_dir = Path(temp_dir) / "profile"
            profile_dir.mkdir()
            port = _reserve_port()

            async with async_playwright() as playwright:
                executable = account_recovery._browser_executable(
                    {}, playwright.chromium.executable_path
                )
                process = subprocess.Popen(
                    [
                        executable,
                        f"--remote-debugging-port={port}",
                        "--remote-allow-origins=*",
                        f"--user-data-dir={profile_dir}",
                        "--headless=new",
                        "--no-first-run",
                        "--no-default-browser-check",
                        "about:blank",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                session = None
                try:
                    await _wait_for_cdp(port)
                    session = await account_recovery._acquire_browser_session(
                        playwright,
                        {
                            "chrome_fetch_reuse_cdp": True,
                            "chrome_fetch_cdp_port": port,
                            "chrome_fetch_executable": executable,
                        },
                        profile_dir=profile_dir,
                        headless=True,
                    )
                    self.assertEqual(session.mode, "borrowed_cdp")

                    await session.context.add_cookies(
                        [
                            {
                                "name": "arena-auth-prod-v1",
                                "value": _authenticated_cookie(),
                                "domain": ".arena.ai",
                                "path": "/",
                            }
                        ]
                    )
                    _, inspection, _ = await account_recovery._current_context_auth(
                        session.context
                    )
                    self.assertTrue(inspection.authenticated)

                    await session.close()
                    await _wait_for_cdp(port, timeout=3)
                    page = await session.context.new_page()
                    await page.close()
                finally:
                    if session is not None and session.browser is not None:
                        try:
                            await session.browser.close()
                        except Exception:
                            pass
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()

    async def test_owned_persistent_context_is_closed_by_session(self):
        with tempfile.TemporaryDirectory(prefix="lmarena-owned-test-") as temp_dir:
            profile_dir = Path(temp_dir) / "profile"
            async with async_playwright() as playwright:
                session = await account_recovery._acquire_browser_session(
                    playwright,
                    {"chrome_fetch_reuse_cdp": False},
                    profile_dir=profile_dir,
                    headless=True,
                )
                self.assertEqual(session.mode, "owned_persistent")
                await session.close()
                with self.assertRaises(Exception):
                    await session.context.new_page()
