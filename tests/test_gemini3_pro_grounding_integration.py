import json
import os
import tempfile
import unittest
from pathlib import Path

import httpx


class TestGemini3ProGroundingIntegration(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(
        os.getenv("RUN_LMARENA_INTEGRATION") == "1",
        "Set RUN_LMARENA_INTEGRATION=1 to run this real external integration test.",
    )
    async def test_bridge_streams_gemini_3_pro_grounding(self) -> None:
        from src import main

        model_name = os.getenv("LMARENA_TEST_MODEL", "gemini-3-pro-grounding")

        source_config_path = Path("config.json")
        if not source_config_path.exists():
            self.skipTest("No config.json found (need existing bridge config with auth tokens).")

        source_cfg = json.loads(source_config_path.read_text(encoding="utf-8"))
        auth_tokens = source_cfg.get("auth_tokens") or []
        if not auth_tokens:
            self.skipTest("No auth_tokens found in config.json (login via dashboard first).")

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        test_config_path = Path(temp_dir.name) / "config.json"

        test_cfg = {
            "password": "admin",
            "auth_tokens": auth_tokens,
            "cf_clearance": source_cfg.get("cf_clearance", ""),
            "cf_bm": source_cfg.get("cf_bm", ""),
            "cfuvid": source_cfg.get("cfuvid", ""),
            "provisional_user_id": source_cfg.get("provisional_user_id", ""),
            "user_agent": source_cfg.get("user_agent", ""),
            "browser_cookies": source_cfg.get("browser_cookies", {}),
            "recaptcha_headless": source_cfg.get("recaptcha_headless", False),
            "api_keys": [
                {
                    "name": "integration",
                    "key": "test-key",
                    "rpm": 999,
                }
            ],
        }
        test_config_path.write_text(json.dumps(test_cfg), encoding="utf-8")

        original_config_file = main.CONFIG_FILE
        main.CONFIG_FILE = str(test_config_path)
        self.addCleanup(setattr, main, "CONFIG_FILE", original_config_file)

        main.chat_sessions.clear()
        main.api_key_usage.clear()

        transport = httpx.ASGITransport(app=main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.stream(
                "POST",
                "/api/v1/chat/completions",
                headers={"Authorization": "Bearer test-key"},
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
                timeout=300.0,
            ) as response:
                body_bytes = await response.aread()

        body_text = body_bytes.decode("utf-8", errors="replace")
        data_lines = [
            line[len("data:") :].strip()
            for line in body_text.splitlines()
            if line.startswith("data:")
        ]

        errors = []
        content_parts = []
        saw_done = False
        for item in data_lines:
            if item == "[DONE]":
                saw_done = True
                continue
            try:
                payload = json.loads(item)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict) and "error" in payload:
                errors.append(payload["error"])
                continue

            if isinstance(payload, dict):
                choices = payload.get("choices") or []
                if choices:
                    delta = (choices[0] or {}).get("delta") or {}
                    if "content" in delta:
                        content_parts.append(delta["content"])

        self.assertTrue(saw_done, msg=f"Stream did not finish. First 500 chars:\n{body_text[:500]}")
        self.assertFalse(errors, msg=f"Upstream error: {errors[0] if errors else None}")
        self.assertTrue(
            "".join(content_parts).strip(),
            msg=f"No content received. First 500 chars:\n{body_text[:500]}",
        )
