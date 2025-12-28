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


class TestDashboardFlow(unittest.TestCase):
    def setUp(self):
        self._orig_models_json = None
        if os.path.exists("models.json"):
            with open("models.json", "r", encoding="utf-8") as f:
                self._orig_models_json = f.read()

        self.config = {
            "api_keys": [],
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
            }
        ]

        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f)
        with open("models.json", "w", encoding="utf-8") as f:
            json.dump(self.models, f)

        main.dashboard_sessions.clear()
        main.api_key_usage.clear()
        main.model_usage_stats.clear()

        # Speed up startup (no browser work)
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

    def test_login_and_manage_keys(self):
        with TestClient(main.app) as client:
            # Unauthed dashboard redirects to login
            resp = client.get("/dashboard", follow_redirects=False)
            self.assertIn(resp.status_code, (302, 303, 307))

            # Login
            resp = client.post("/login", data={"password": "admin"}, follow_redirects=False)
            self.assertEqual(resp.status_code, 303)
            session_id = resp.cookies.get("session_id")
            self.assertTrue(session_id)
            client.cookies.set("session_id", session_id)

            # Dashboard renders
            resp = client.get("/dashboard")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("Dashboard", resp.text)

            # Create key
            resp = client.post(
                "/create-key",
                data={"name": "k1", "rpm": 10},
                follow_redirects=False,
            )
            self.assertEqual(resp.status_code, 303)

            with open("config.json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.assertEqual(len(cfg["api_keys"]), 1)
            key_id = cfg["api_keys"][0]["key"]

            # Delete key
            resp = client.post(
                "/delete-key",
                data={"key_id": key_id},
                follow_redirects=False,
            )
            self.assertEqual(resp.status_code, 303)
            with open("config.json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.assertEqual(cfg["api_keys"], [])

            # Add auth token
            resp = client.post(
                "/add-auth-token",
                data={"new_auth_token": "extra-token"},
                follow_redirects=False,
            )
            self.assertEqual(resp.status_code, 303)
            with open("config.json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.assertIn("extra-token", cfg["auth_tokens"])

            # Delete auth token at index 1
            resp = client.post(
                "/delete-auth-token",
                data={"token_index": 1},
                follow_redirects=False,
            )
            self.assertEqual(resp.status_code, 303)
            with open("config.json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.assertNotIn("extra-token", cfg["auth_tokens"])

            # Update primary auth token
            resp = client.post(
                "/update-auth-token",
                data={"auth_token": "updated"},
                follow_redirects=False,
            )
            self.assertEqual(resp.status_code, 303)
            with open("config.json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.assertEqual(cfg["auth_token"], "updated")

            # Refresh tokens endpoint (patched to no-op)
            resp = client.post("/refresh-tokens", follow_redirects=False)
            self.assertEqual(resp.status_code, 303)

    def test_login_wrong_password_redirects(self):
        with TestClient(main.app) as client:
            resp = client.post("/login", data={"password": "wrong"}, follow_redirects=False)

        self.assertEqual(resp.status_code, 303)
        self.assertIn("/login?error=1", resp.headers.get("location", ""))

    def test_logout_clears_session(self):
        with TestClient(main.app) as client:
            resp = client.post("/login", data={"password": "admin"}, follow_redirects=False)
            session_id = resp.cookies.get("session_id")
            self.assertTrue(session_id)
            self.assertIn(session_id, main.dashboard_sessions)

            client.cookies.set("session_id", session_id)
            resp = client.get("/logout", follow_redirects=False)

        self.assertEqual(resp.status_code, 303)
        self.assertEqual(resp.headers.get("location"), "/login")
        self.assertNotIn(session_id, main.dashboard_sessions)

    def test_dashboard_renders_keys_and_stats(self):
        # Seed one API key so keys table loop executes.
        self.config["api_keys"] = [
            {"name": "k1", "key": "k1-key", "rpm": 10, "created": 1700000000}
        ]
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f)

        # Seed usage stats so stats table loop executes (startup persists these to config.json).
        main.model_usage_stats["gpt-4"] = 3

        with TestClient(main.app) as client:
            resp = client.post("/login", data={"password": "admin"}, follow_redirects=False)
            session_id = resp.cookies.get("session_id")
            client.cookies.set("session_id", session_id)

            resp = client.get("/dashboard")

        self.assertEqual(resp.status_code, 200)
        self.assertIn("k1", resp.text)
        self.assertIn("gpt-4", resp.text)
        self.assertIn("3", resp.text)

    def test_dashboard_returns_error_page_when_config_fails(self):
        with TestClient(main.app) as client:
            resp = client.post("/login", data={"password": "admin"}, follow_redirects=False)
            session_id = resp.cookies.get("session_id")
            client.cookies.set("session_id", session_id)

            with patch.object(main, "get_config", side_effect=RuntimeError("boom")):
                resp = client.get("/dashboard")

        self.assertEqual(resp.status_code, 500)
        self.assertIn("Dashboard Error", resp.text)
