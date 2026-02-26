import json
from pathlib import Path

from tests._stream_test_utils import BaseBridgeTest


class TestApiKeyPersistence(BaseBridgeTest):
    async def test_save_config_preserves_api_keys_from_disk_by_default(self) -> None:
        config = self.main.get_config()
        config["cf_clearance"] = "cf"
        config["api_keys"] = []

        self.main.save_config(config)

        on_disk = json.loads(Path(self.main.CONFIG_FILE).read_text(encoding="utf-8"))
        self.assertEqual(on_disk["api_keys"][0]["key"], "test-key")

    async def test_save_config_allows_api_key_updates_when_disabled(self) -> None:
        config = self.main.get_config()
        config["api_keys"].append(
            {"name": "New Key", "key": "new-key", "rpm": 1, "created": 1},
        )

        self.main.save_config(config, preserve_api_keys=False)

        on_disk = json.loads(Path(self.main.CONFIG_FILE).read_text(encoding="utf-8"))
        keys = [entry.get("key") for entry in on_disk.get("api_keys", [])]
        self.assertIn("new-key", keys)

    async def test_save_config_preserves_empty_api_key_list_from_disk(self) -> None:
        self.setup_config({"api_keys": []})

        config = self.main.get_config()
        config["api_keys"] = [{"name": "Generated Key", "key": "generated-key", "rpm": 1, "created": 1}]

        self.main.save_config(config)

        on_disk = json.loads(Path(self.main.CONFIG_FILE).read_text(encoding="utf-8"))
        self.assertEqual(on_disk.get("api_keys"), [])
