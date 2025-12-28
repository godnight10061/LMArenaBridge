import base64
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


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, text: str = "", headers: dict | None = None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("HTTP error")


class _FakeAsyncClient:
    def __init__(self, responses: list[_FakeResponse]):
        self._responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return self._responses.pop(0)

    async def put(self, *args, **kwargs):
        return self._responses.pop(0)


class TestImageHelpers(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
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
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f)

    async def asyncTearDown(self):
        if os.path.exists("config.json"):
            os.remove("config.json")

    async def test_upload_image_to_lmarena_success(self):
        upload_payload = {
            "success": True,
            "data": {"uploadUrl": "https://upload.example", "key": "key123"},
        }
        download_payload = {
            "success": True,
            "data": {"url": "https://download.example/file"},
        }
        responses = [
            _FakeResponse(text=f'0:null\n1:{json.dumps(upload_payload)}\n'),
            _FakeResponse(),  # PUT upload
            _FakeResponse(text=f'0:null\n1:{json.dumps(download_payload)}\n'),
        ]

        with patch("httpx.AsyncClient", return_value=_FakeAsyncClient(responses)):
            result = await main.upload_image_to_lmarena(b"img", "image/png", "x.png")

        self.assertEqual(result, ("key123", "https://download.example/file"))

    async def test_upload_image_to_lmarena_invalid_mime_type_returns_none(self):
        result = await main.upload_image_to_lmarena(b"img", "text/plain", "x.txt")
        self.assertIsNone(result)

    async def test_upload_image_to_lmarena_missing_next_action_ids_returns_none(self):
        # Remove action IDs from config
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump({**self.config, "next_action_upload": "", "next_action_signed_url": ""}, f)

        result = await main.upload_image_to_lmarena(b"img", "image/png", "x.png")
        self.assertIsNone(result)

    async def test_upload_image_to_lmarena_timeout_returns_none(self):
        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *args, **kwargs):
                raise main.httpx.TimeoutException("timeout")

        with patch.object(main.httpx, "AsyncClient", return_value=_Client()):
            result = await main.upload_image_to_lmarena(b"img", "image/png", "x.png")

        self.assertIsNone(result)

    async def test_process_message_content_uploads_image_when_supported(self):
        raw = b"img"
        data_uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")

        model_capabilities = {"inputCapabilities": {"image": True}}
        content = [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]

        with patch.object(main, "upload_image_to_lmarena", new=AsyncMock(return_value=("k", "https://d"))):
            text, attachments = await main.process_message_content(content, model_capabilities)

        self.assertEqual(text, "hi")
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0]["name"], "k")
        self.assertEqual(attachments[0]["url"], "https://d")

    async def test_process_message_content_ignores_images_when_not_supported(self):
        model_capabilities = {"inputCapabilities": {"image": False}}
        content = [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ]

        with patch.object(main, "upload_image_to_lmarena", new=AsyncMock()) as upload:
            text, attachments = await main.process_message_content(content, model_capabilities)

        self.assertEqual(text, "hi")
        self.assertEqual(attachments, [])
        upload.assert_not_awaited()

    async def test_process_message_content_skips_invalid_data_uri(self):
        model_capabilities = {"inputCapabilities": {"image": True}}
        content = [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64AA=="}},  # missing comma
        ]

        with patch.object(main, "upload_image_to_lmarena", new=AsyncMock()) as upload:
            text, attachments = await main.process_message_content(content, model_capabilities)

        self.assertEqual(text, "hi")
        self.assertEqual(attachments, [])
        upload.assert_not_awaited()

    async def test_process_message_content_skips_too_large_image(self):
        model_capabilities = {"inputCapabilities": {"image": True}}
        content = [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ]

        with patch.object(main.base64, "b64decode", return_value=b"0" * (10 * 1024 * 1024 + 1)), patch.object(
            main, "upload_image_to_lmarena", new=AsyncMock()
        ) as upload:
            text, attachments = await main.process_message_content(content, model_capabilities)

        self.assertEqual(text, "hi")
        self.assertEqual(attachments, [])
        upload.assert_not_awaited()

    async def test_process_message_content_skips_base64_decode_error(self):
        model_capabilities = {"inputCapabilities": {"image": True}}
        content = [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ]

        with patch.object(main.base64, "b64decode", side_effect=Exception("bad base64")), patch.object(
            main, "upload_image_to_lmarena", new=AsyncMock()
        ) as upload:
            text, attachments = await main.process_message_content(content, model_capabilities)

        self.assertEqual(text, "hi")
        self.assertEqual(attachments, [])
        upload.assert_not_awaited()

    async def test_process_message_content_skips_external_url(self):
        model_capabilities = {"inputCapabilities": {"image": True}}
        content = [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]

        with patch.object(main, "upload_image_to_lmarena", new=AsyncMock()) as upload:
            text, attachments = await main.process_message_content(content, model_capabilities)

        self.assertEqual(text, "hi")
        self.assertEqual(attachments, [])
        upload.assert_not_awaited()
