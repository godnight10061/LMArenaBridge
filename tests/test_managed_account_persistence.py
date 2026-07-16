import base64
import json
import tempfile
import time
import unittest
from pathlib import Path

from src import account_recovery, main


def _authenticated_cookie() -> str:
    payload = {
        "expires_at": int(time.time()) + 3600,
        "user": {
            "aud": "authenticated",
            "role": "authenticated",
            "email": "managed@example.invalid",
            "is_anonymous": False,
        },
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return "base64-" + encoded.rstrip("=")


class ManagedAccountPersistenceTests(unittest.TestCase):
    def test_managed_recovery_saver_persists_verified_token(self) -> None:
        original_config_file = main.CONFIG_FILE
        runtime = tempfile.TemporaryDirectory(prefix="managed-account-persistence-")
        try:
            config_path = Path(runtime.name) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "auth_token": "",
                        "auth_tokens": [],
                        "api_keys": [],
                        "managed_account": {},
                        "managed_account_history": [],
                    }
                ),
                encoding="utf-8",
            )
            main.CONFIG_FILE = str(config_path)
            token = _authenticated_cookie()

            account_recovery._persist(
                main.get_config,
                main._save_managed_account_config,
                token=token,
                cookies=[
                    {
                        "name": "arena-auth-prod-v1",
                        "value": token,
                        "domain": ".arena.ai",
                        "path": "/",
                    }
                ],
                profile_dir=Path(runtime.name) / "chrome-profile",
                executable="chrome.exe",
                account={
                    "schema": 1,
                    "provider": "mail.tm",
                    "address": "managed@example.invalid",
                    "password": "fixture-password",
                    "created_at": int(time.time()),
                    "last_verified_at": int(time.time()),
                    "status": "active",
                },
            )

            saved = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["auth_token"], token)
            self.assertEqual(saved["auth_tokens"], [token])
            self.assertEqual(saved["managed_account"]["provider"], "mail.tm")
        finally:
            main.CONFIG_FILE = original_config_file
            runtime.cleanup()


if __name__ == "__main__":
    unittest.main()
