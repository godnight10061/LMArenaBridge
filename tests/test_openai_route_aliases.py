import json
import tempfile
import unittest
from pathlib import Path

import httpx

from src import main


class OpenAIRouteAliasTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._original_config_file = main.CONFIG_FILE
        self._original_debug = main.DEBUG
        self._runtime = tempfile.TemporaryDirectory(prefix="openai-route-alias-test-")
        config_path = Path(self._runtime.name) / "config.json"
        config_path.write_text(
            json.dumps({"api_keys": [], "auth_tokens": []}),
            encoding="utf-8",
        )
        main.CONFIG_FILE = str(config_path)
        main.DEBUG = False
        main.api_key_usage.clear()

    async def asyncTearDown(self) -> None:
        main.CONFIG_FILE = self._original_config_file
        main.DEBUG = self._original_debug
        main.api_key_usage.clear()
        self._runtime.cleanup()

    def test_chat_completion_paths_share_one_handler(self) -> None:
        post_routes = {
            route.path: route.endpoint
            for route in main.app.routes
            if "POST" in getattr(route, "methods", set())
        }
        get_routes = {
            route.path: route.endpoint
            for route in main.app.routes
            if "GET" in getattr(route, "methods", set())
        }

        self.assertIn("/v1/chat/completions", post_routes)
        self.assertIn("/api/v1/chat/completions", post_routes)
        self.assertIs(
            post_routes["/v1/chat/completions"],
            post_routes["/api/v1/chat/completions"],
        )
        self.assertIs(post_routes["/v1/chat/completions"], main.api_chat_completions)
        self.assertIs(get_routes["/v1/models"], get_routes["/api/v1/models"])
        self.assertIs(get_routes["/v1/models"], main.list_models)
        self.assertIs(get_routes["/v1/health"], get_routes["/api/v1/health"])
        self.assertIs(get_routes["/v1/health"], main.health_check)

    async def test_both_paths_reach_identical_handler_validation(self) -> None:
        transport = httpx.ASGITransport(app=main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://bridge.test") as client:
            standard = await client.post("/v1/chat/completions", json={})
            compatibility = await client.post("/api/v1/chat/completions", json={})

        self.assertEqual(standard.status_code, 400)
        self.assertEqual(compatibility.status_code, 400)
        self.assertEqual(standard.json(), compatibility.json())
        self.assertEqual(standard.json()["detail"], "Missing 'model' in request body.")


if __name__ == "__main__":
    unittest.main()
