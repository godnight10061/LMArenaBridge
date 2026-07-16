import json
import tempfile
import unittest
from pathlib import Path

from src import main


def _models(count: int = 5):
    return [
        {
            "id": f"id-{index}",
            "publicName": f"model-{index}",
            "organization": f"provider-{index}",
            "capabilities": {"outputCapabilities": {"text": True}},
        }
        for index in range(count)
    ]


class ModelCatalogComponentTests(unittest.TestCase):
    def test_browser_ui_routing_covers_required_and_configured_text_models(self):
        self.assertTrue(
            main._should_use_browser_ui(
                "gemini-3.5-flash", "chat", all_text_models=False
            )
        )
        self.assertTrue(
            main._should_use_browser_ui("gpt-4", "chat", all_text_models=True)
        )
        self.assertFalse(
            main._should_use_browser_ui("gpt-4", "chat", all_text_models=False)
        )
        self.assertFalse(
            main._should_use_browser_ui("gpt-4", "search", all_text_models=True)
        )

    def test_extracts_and_validates_live_initial_models(self):
        payload = json.dumps(_models(), separators=(",", ":"))
        escaped = payload.replace('"', '\\"')
        html = '{\\"initialModels\\":' + escaped + ',\\"initialModelAId\\":\\"x\\"}'

        result = main._extract_live_model_catalog(html)

        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]["publicName"], "model-0")

    def test_rejects_catalog_without_five_organized_models(self):
        with self.assertRaises(ValueError):
            main._validate_live_model_catalog(_models(4))

    def test_atomic_save_replaces_catalog_without_temp_files(self):
        original = main.MODELS_FILE
        with tempfile.TemporaryDirectory(prefix="model-catalog-test-") as temp_dir:
            path = Path(temp_dir) / "models.json"
            try:
                main.MODELS_FILE = str(path)
                self.assertTrue(main.save_models(_models()))
                self.assertEqual(len(main.get_models()), 5)
                self.assertEqual(list(path.parent.glob(".*.tmp")), [])
            finally:
                main.MODELS_FILE = original

    def test_catalog_status_survives_followup_config_write(self):
        original_config = main.CONFIG_FILE
        with tempfile.TemporaryDirectory(prefix="model-status-test-") as temp_dir:
            path = Path(temp_dir) / "config.json"
            try:
                main.CONFIG_FILE = str(path)
                main.save_config({"api_keys": [], "model_catalog": {}})
                main._record_model_catalog_status(fresh=True, count=7)
                followup = main.get_config()
                followup["next_action_upload"] = "action-id"
                main.save_config(followup)

                saved = main.get_config()["model_catalog"]
                self.assertTrue(saved["fresh"])
                self.assertEqual(saved["count"], 7)
            finally:
                main.CONFIG_FILE = original_config


if __name__ == "__main__":
    unittest.main()
