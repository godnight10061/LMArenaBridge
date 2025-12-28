import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path so we can import main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Avoid importing heavy browser deps during unit tests
with patch.dict(sys.modules, {"camoufox.async_api": MagicMock(), "camoufox": MagicMock()}):
    import main

from fastapi.testclient import TestClient


class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        self._orig_models_json = None
        if os.path.exists("models.json"):
            with open("models.json", "r", encoding="utf-8") as f:
                self._orig_models_json = f.read()

        self.config = {
            "api_keys": [
                {"name": "Test Key", "key": "test-key", "rpm": 1000, "created": 1700000000}
            ],
            "auth_tokens": ["test-token"],
            "cf_clearance": "test-clearance",
            "password": "admin",
            "auth_token": "test-token",
            "usage_stats": {},
            "next_action_upload": "upload-action",
            "next_action_signed_url": "signed-url-action",
        }
        self.models = [
            {
                "id": "gpt-4",
                "publicName": "gpt-4",
                "organization": "openai",
                "capabilities": {"outputCapabilities": {"text": True}},
            },
            {
                "id": "stealth",
                "publicName": "stealth",
                "organization": "",
                "capabilities": {"outputCapabilities": {"text": True}},
            },
        ]

        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f)
        with open("models.json", "w", encoding="utf-8") as f:
            json.dump(self.models, f)

        # Reset mutable globals between tests
        main.chat_sessions.clear()
        main.dashboard_sessions.clear()
        main.api_key_usage.clear()
        main.model_usage_stats.clear()

        # Speed up startup in TestClient
        self.initial_data_patcher = patch.object(main, "get_initial_data", new=AsyncMock(return_value=None))
        self.periodic_refresh_patcher = patch.object(main, "periodic_refresh_task", new=AsyncMock(return_value=None))
        self.recaptcha_patcher = patch.object(main, "refresh_recaptcha_token", new=AsyncMock(return_value="token"))
        self.initial_data_patcher.start()
        self.periodic_refresh_patcher.start()
        self.recaptcha_patcher.start()

    def tearDown(self):
        self.initial_data_patcher.stop()
        self.periodic_refresh_patcher.stop()
        self.recaptcha_patcher.stop()

        if os.path.exists("config.json"):
            os.remove("config.json")

        if self._orig_models_json is not None:
            with open("models.json", "w", encoding="utf-8") as f:
                f.write(self._orig_models_json)
        elif os.path.exists("models.json"):
            os.remove("models.json")

    def test_models_requires_bearer_prefix(self):
        with TestClient(main.app) as client:
            resp = client.get("/api/v1/models", headers={"Authorization": "test-key"})
        self.assertEqual(resp.status_code, 401)

    def test_models_list(self):
        with TestClient(main.app) as client:
            resp = client.get("/api/v1/models", headers={"Authorization": "Bearer test-key"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["object"], "list")
        ids = [m["id"] for m in body["data"]]
        self.assertIn("gpt-4", ids)

    def test_chat_missing_model_returns_400(self):
        with TestClient(main.app) as client:
            resp = client.post(
                "/api/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer test-key"},
            )
        self.assertEqual(resp.status_code, 400)

    def test_chat_unknown_model_returns_404(self):
        with TestClient(main.app) as client:
            resp = client.post(
                "/api/v1/chat/completions",
                json={"model": "unknown-model", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer test-key"},
            )
        self.assertEqual(resp.status_code, 404)

    def test_chat_stealth_model_returns_403(self):
        with TestClient(main.app) as client:
            resp = client.post(
                "/api/v1/chat/completions",
                json={"model": "stealth", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer test-key"},
            )
        self.assertEqual(resp.status_code, 403)

    def test_health_endpoint(self):
        with TestClient(main.app) as client:
            resp = client.get("/api/v1/health")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(resp.json()["status"], {"healthy", "degraded"})

    def test_rate_limiting_returns_429(self):
        # Set RPM to 1 and hit an authenticated endpoint twice.
        config = dict(self.config)
        config["api_keys"] = [dict(config["api_keys"][0])]
        config["api_keys"][0]["rpm"] = 1
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        with TestClient(main.app) as client:
            r1 = client.get("/api/v1/models", headers={"Authorization": "Bearer test-key"})
            r2 = client.get("/api/v1/models", headers={"Authorization": "Bearer test-key"})

        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 429)
