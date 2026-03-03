import json
from pathlib import Path

import httpx

from tests._stream_test_utils import BaseBridgeTest


class TestDashboardSecurity(BaseBridgeTest):
    async def _login(self, client: httpx.AsyncClient) -> str:
        resp = await client.post("/login", data={"password": "admin"}, follow_redirects=False)
        self.assertIn(resp.status_code, (302, 303))
        session_id = client.cookies.get("session_id")
        self.assertTrue(session_id)
        return str(session_id)

    async def test_dashboard_escapes_untrusted_values_and_includes_csrf(self) -> None:
        self.setup_config(
            {
                "api_keys": [
                    {
                        "name": '<img src=x onerror=alert(1)>',
                        "key": "test-key",
                        "rpm": 60,
                        "created": 1,
                    }
                ]
            }
        )
        self.main.model_usage_stats.clear()
        self.main.model_usage_stats["</script><script>alert(1)</script>"] = 1

        transport = httpx.ASGITransport(app=self.main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await self._login(client)
            response = await client.get("/dashboard")

        self.assertEqual(response.status_code, 200)
        text = response.text

        self.assertNotIn('<img src=x onerror=alert(1)>', text)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", text)

        self.assertNotIn("</script><script>alert(1)</script>", text)
        self.assertIn("&lt;/script&gt;&lt;script&gt;alert(1)&lt;/script&gt;", text)

        self.assertIn('name="csrf_token"', text)

    async def test_create_key_requires_csrf(self) -> None:
        transport = httpx.ASGITransport(app=self.main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            session_id = await self._login(client)

            missing = await client.post("/create-key", data={"name": "New Key", "rpm": "60"})
            self.assertEqual(missing.status_code, 403)

            csrf = self.main.dashboard_sessions[session_id]["csrf_token"]
            created = await client.post(
                "/create-key",
                data={"name": "New Key", "rpm": "60", "csrf_token": csrf},
                follow_redirects=False,
            )
            self.assertIn(created.status_code, (302, 303))

        on_disk = json.loads(Path(self.main.CONFIG_FILE).read_text(encoding="utf-8"))
        names = [entry.get("name") for entry in on_disk.get("api_keys", [])]
        self.assertIn("New Key", names)

    async def test_delete_key_requires_csrf(self) -> None:
        transport = httpx.ASGITransport(app=self.main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            session_id = await self._login(client)
            csrf = self.main.dashboard_sessions[session_id]["csrf_token"]

            missing = await client.post("/delete-key", data={"key_id": "test-key", "csrf_token": "bad"})
            self.assertEqual(missing.status_code, 403)

            deleted = await client.post(
                "/delete-key",
                data={"key_id": "test-key", "csrf_token": csrf},
                follow_redirects=False,
            )
            self.assertIn(deleted.status_code, (302, 303))

        on_disk = json.loads(Path(self.main.CONFIG_FILE).read_text(encoding="utf-8"))
        keys = [entry.get("key") for entry in on_disk.get("api_keys", [])]
        self.assertNotIn("test-key", keys)

