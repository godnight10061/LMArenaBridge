import asyncio
import json
import os
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
            main._should_use_browser_ui("gemini-3.5-flash", "chat", all_text_models=False)
        )
        self.assertTrue(main._should_use_browser_ui("gpt-4", "chat", all_text_models=True))
        self.assertFalse(main._should_use_browser_ui("gpt-4", "chat", all_text_models=False))
        self.assertFalse(main._should_use_browser_ui("gpt-4", "search", all_text_models=True))

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

    def test_normalizes_internal_variants_to_preferred_public_model(self):
        models = _models()
        models.extend(
            [
                {
                    "id": "text-only",
                    "publicName": "gemini-3.5-flash",
                    "organization": "google",
                    "userSelectable": True,
                    "capabilities": {
                        "inputCapabilities": {"text": True},
                        "outputCapabilities": {"text": True},
                    },
                },
                {
                    "id": "preferred",
                    "publicName": "gemini-3.5-flash",
                    "organization": "google",
                    "provider": "googleVertexGlobal",
                    "name": "gemini-3.5-flash",
                    "userSelectable": True,
                    "capabilities": {
                        "inputCapabilities": {
                            "text": True,
                            "image": True,
                            "file": True,
                        },
                        "outputCapabilities": {"text": True, "web": True},
                    },
                },
                {
                    "id": "hidden",
                    "publicName": "hidden-model",
                    "organization": "provider",
                    "userSelectable": False,
                },
            ]
        )

        normalized, eligible_count = main._normalize_public_model_catalog(models)

        self.assertEqual(eligible_count, 7)
        self.assertEqual(len(normalized), len({model["publicName"] for model in normalized}))
        gemini = next(model for model in normalized if model["publicName"] == "gemini-3.5-flash")
        self.assertEqual(gemini["id"], "preferred")
        self.assertNotIn("hidden-model", {model["publicName"] for model in normalized})

    def test_normalizes_legacy_raw_cache_before_degraded_fallback(self):
        cached = _models()
        cached.append(
            {
                "id": "preferred-duplicate",
                "publicName": "model-0",
                "organization": "provider-0",
                "provider": "provider-api",
                "name": "model-0",
                "userSelectable": True,
                "capabilities": {
                    "inputCapabilities": {"text": True, "image": True},
                    "outputCapabilities": {"text": True},
                },
            }
        )

        normalized, raw_count, eligible_count = main._normalize_cached_model_catalog(cached)

        self.assertEqual(raw_count, 6)
        self.assertEqual(eligible_count, 6)
        self.assertEqual(len(normalized), 5)
        self.assertEqual(normalized[0]["id"], "preferred-duplicate")

    def test_rejects_invalid_cached_catalog_shape(self):
        with self.assertRaises(ValueError):
            main._normalize_cached_model_catalog({"legacy": "not-a-list"})

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

    def test_get_models_returns_empty_for_non_list_json(self):
        original = main.MODELS_FILE
        with tempfile.TemporaryDirectory(prefix="model-catalog-shape-test-") as temp_dir:
            path = Path(temp_dir) / "models.json"
            path.write_text('{"legacy":"not-a-list"}', encoding="utf-8")
            try:
                main.MODELS_FILE = str(path)
                self.assertEqual(main.get_models(), [])
            finally:
                main.MODELS_FILE = original

    def test_catalog_setup_failure_uses_normalized_validated_cache(self):
        original_config = main.CONFIG_FILE
        original_models = main.MODELS_FILE
        original_profile = os.environ.get("LM_CHROME_PROFILE_DIR")
        with tempfile.TemporaryDirectory(prefix="model-catalog-fallback-test-") as temp_dir:
            root = Path(temp_dir)
            blocked_profile = root / "profile-is-a-file"
            blocked_profile.write_text("not a directory", encoding="utf-8")
            cached = _models()
            cached.append(
                {
                    "id": "preferred-duplicate",
                    "publicName": "model-0",
                    "organization": "provider-0",
                    "provider": "provider-api",
                    "name": "model-0",
                    "userSelectable": True,
                    "capabilities": {
                        "inputCapabilities": {"text": True, "image": True},
                        "outputCapabilities": {"text": True},
                    },
                }
            )
            try:
                main.CONFIG_FILE = str(root / "config.json")
                main.MODELS_FILE = str(root / "models.json")
                main.save_config({"api_keys": [], "model_catalog": {}})
                main.save_models(cached)
                os.environ["LM_CHROME_PROFILE_DIR"] = str(blocked_profile)

                status = asyncio.run(main.get_initial_data())

                self.assertFalse(status["fresh"])
                self.assertEqual(status["source"], "validated_cache")
                self.assertEqual(status["count"], 5)
                self.assertEqual(status["raw_count"], 6)
                self.assertEqual(status["eligible_count"], 6)
                self.assertEqual(len(main.get_models()), 5)
                self.assertEqual(main.get_models()[0]["id"], "preferred-duplicate")
            finally:
                main.CONFIG_FILE = original_config
                main.MODELS_FILE = original_models
                if original_profile is None:
                    os.environ.pop("LM_CHROME_PROFILE_DIR", None)
                else:
                    os.environ["LM_CHROME_PROFILE_DIR"] = original_profile

    def test_catalog_status_survives_followup_config_write(self):
        original_config = main.CONFIG_FILE
        with tempfile.TemporaryDirectory(prefix="model-status-test-") as temp_dir:
            path = Path(temp_dir) / "config.json"
            try:
                main.CONFIG_FILE = str(path)
                main.save_config({"api_keys": [], "model_catalog": {}})
                main._record_model_catalog_status(
                    fresh=True, count=7, raw_count=12, eligible_count=9
                )
                followup = main.get_config()
                followup["next_action_upload"] = "action-id"
                main.save_config(followup)

                saved = main.get_config()["model_catalog"]
                self.assertTrue(saved["fresh"])
                self.assertEqual(saved["count"], 7)
                self.assertEqual(saved["raw_count"], 12)
                self.assertEqual(saved["eligible_count"], 9)
            finally:
                main.CONFIG_FILE = original_config

    def test_catalog_status_can_report_an_unavailable_cache(self):
        original_config = main.CONFIG_FILE
        with tempfile.TemporaryDirectory(prefix="model-status-source-test-") as temp_dir:
            path = Path(temp_dir) / "config.json"
            try:
                main.CONFIG_FILE = str(path)
                main.save_config({"api_keys": [], "model_catalog": {}})
                main._record_model_catalog_status(
                    fresh=False,
                    count=0,
                    source="unavailable",
                    error_code="chrome_refresh_runtimeerror_cache_invalid",
                )

                saved = main.get_config()["model_catalog"]
                self.assertFalse(saved["fresh"])
                self.assertEqual(saved["source"], "unavailable")
                self.assertEqual(saved["count"], 0)
            finally:
                main.CONFIG_FILE = original_config


if __name__ == "__main__":
    unittest.main()
