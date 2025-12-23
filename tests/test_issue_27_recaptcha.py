import json
import os
import secrets
import time
import unittest
from pathlib import Path

import httpx


LMARENA_ORIGIN = "https://lmarena.ai"


def uuid7() -> str:
    timestamp_ms = int(time.time() * 1000)
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)

    uuid_int = timestamp_ms << 80
    uuid_int |= (0x7000 | rand_a) << 64
    uuid_int |= 0x8000000000000000 | rand_b

    hex_str = f"{uuid_int:032x}"
    return f"{hex_str[0:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"


class TestIssue27RecaptchaValidation(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(
        os.getenv("RUN_LMARENA_INTEGRATION") == "1",
        "Set RUN_LMARENA_INTEGRATION=1 to run this real external integration test.",
    )
    async def test_create_evaluation_does_not_fail_recaptcha(self) -> None:
        cfg_path = Path("config.json")
        if not cfg_path.exists():
            self.skipTest("No config.json found (need existing bridge config with auth tokens).")

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        tokens = cfg.get("auth_tokens") or []
        if not tokens:
            self.skipTest("No auth_tokens found in config.json (login via dashboard first).")

        auth_token = tokens[0]
        from src import main

        if not main.find_chrome_executable(main.get_config()):
            self.skipTest("No Chrome/Edge executable found (required for current LMArena reCAPTCHA flow).")

        models_path = Path("models.json")
        models = json.loads(models_path.read_text(encoding="utf-8")) if models_path.exists() else []
        self.assertTrue(models, "No models.json available (run the bridge once to fetch models).")
        model = next(
            (
                m
                for m in models
                if m.get("publicName") == "gemini-3-pro-grounding"
                or m.get("name") == "gemini-3-pro-grounding"
            ),
            models[0],
        )
        model_id = model["id"]
        capabilities = model.get("capabilities") or {}
        output_caps = (capabilities.get("outputCapabilities") or {}) if isinstance(capabilities, dict) else {}
        modality = "search" if output_caps.get("search") else "chat"

        async def post_create_evaluation(first_payload: dict) -> dict:
            headers = main.get_request_headers_with_token(auth_token)
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{LMARENA_ORIGIN}/nextjs-api/stream/create-evaluation",
                    headers=headers,
                    content=json.dumps(first_payload),
                    timeout=120.0,
                ) as response:
                    first = b""
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            first = chunk
                            break
            return {"status": response.status_code, "first": first.decode("utf-8", errors="replace")}

        recaptcha_token = await main.refresh_recaptcha_token(auth_token=auth_token, force_new=True)
        self.assertIsNotNone(recaptcha_token, "Failed to acquire reCAPTCHA token")
        self.assertGreater(len(recaptcha_token), 200)

        payload = {
            "id": uuid7(),
            "mode": "direct",
            "modelAId": model_id,
            "userMessageId": uuid7(),
            "modelAMessageId": uuid7(),
            "userMessage": {
                "content": "Hello from integration test",
                "experimental_attachments": [],
                "metadata": {},
            },
            "modality": modality,
            "recaptchaV3Token": recaptcha_token,
        }

        result = await post_create_evaluation(payload)

        if result["status"] == 401 and "user not found" in result["first"].lower():
            signed_up = await main.signup_user_if_needed(auth_token)
            self.assertTrue(signed_up, msg=f"Sign-up flow failed: {result['first'][:200]}")
            recaptcha_token = await main.refresh_recaptcha_token(auth_token=auth_token, force_new=True)
            self.assertIsNotNone(recaptcha_token, "Failed to acquire reCAPTCHA token after sign-up")
            payload["recaptchaV3Token"] = recaptcha_token
            result = await post_create_evaluation(payload)

        self.assertEqual(
            result["status"],
            200,
            msg=f"Expected 200 OK but got {result['status']} with first chunk: {result['first'][:200]}",
        )
        self.assertNotIn("recaptcha validation failed", result["first"])
