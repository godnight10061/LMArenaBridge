import json
import tempfile
import unittest
from pathlib import Path


class TestAuthTokenBackCompat(unittest.TestCase):
    def test_single_auth_token_populates_auth_tokens_list(self) -> None:
        from src import main

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config_path = Path(temp_dir.name) / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "password": "admin",
                    "auth_token": "arena-auth-prod-v1-token",
                    "auth_tokens": [],
                    "api_keys": [],
                }
            ),
            encoding="utf-8",
        )

        original_config_file = main.CONFIG_FILE
        main.CONFIG_FILE = str(config_path)
        self.addCleanup(setattr, main, "CONFIG_FILE", original_config_file)

        cfg = main.get_config()
        self.assertEqual(cfg.get("auth_tokens"), ["arena-auth-prod-v1-token"])


if __name__ == "__main__":
    unittest.main()

