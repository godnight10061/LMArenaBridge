import json
import tempfile
import unittest
from pathlib import Path

import httpx


class TestDashboardApiKeySchema(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from src import main

        self.main = main
        self.main.dashboard_sessions.clear()

        self._temp_dir = tempfile.TemporaryDirectory()
        self._config_path = Path(self._temp_dir.name) / "config.json"
        self._config_path.write_text(
            json.dumps(
                {
                    "password": "admin",
                    "api_keys": [
                        {
                            "key": "sk-test-key",
                            "rpm": 60,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        self._orig_config_file = self.main.CONFIG_FILE
        self.main.CONFIG_FILE = str(self._config_path)

    async def asyncTearDown(self) -> None:
        self.main.CONFIG_FILE = self._orig_config_file
        self._temp_dir.cleanup()

    async def test_dashboard_renders_api_key_without_name(self) -> None:
        transport = httpx.ASGITransport(app=self.main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/login",
                data={"password": "admin"},
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("sk-test-key", response.text)
