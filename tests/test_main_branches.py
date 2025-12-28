import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os
import json
import time
from collections import defaultdict

# Add src to path so we can import main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fastapi.testclient import TestClient
from fastapi import HTTPException

# Mock heavy imports
with patch.dict(sys.modules, {'camoufox.async_api': MagicMock(), 'camoufox': MagicMock()}):
    import main

class TestMainBranches(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Setup config
        self.config = {
            "api_keys": [
                {
                    "name": "Test Key",
                    "key": "test-key",
                    "rpm": 60,
                    "created": 1700000000
                }
            ],
            "auth_tokens": ["token1", "token2"],
            "cf_clearance": "clearance",
            "password": "admin",
            "auth_token": "token1",
            "usage_stats": {},
            "next_action_upload": "upload_id",
            "next_action_signed_url": "signed_id"
        }
        
        # Setup models
        self.models = [
            {
                "id": "gpt-4-vision",
                "publicName": "gpt-4-vision",
                "organization": "openai",
                "capabilities": {
                    "inputCapabilities": {"image": True},
                    "outputCapabilities": {"text": True, "image": True}
                }
            },
            {
                "id": "gpt-4",
                "publicName": "gpt-4",
                "organization": "openai",
                "capabilities": {
                    "inputCapabilities": {"image": False},
                    "outputCapabilities": {"text": True}
                }
            }
        ]

        # Manual patching for reliability
        self.original_get_config = main.get_config
        self.original_get_models = main.get_models
        
        main.get_config = MagicMock(return_value=self.config)
        main.get_models = MagicMock(return_value=self.models)
        
        # Reset globals
        main.model_usage_stats = defaultdict(int)
        main.api_key_usage = defaultdict(list)
        main.chat_sessions = defaultdict(dict)
        main.current_token_index = 0

    def tearDown(self):
        main.get_config = self.original_get_config
        main.get_models = self.original_get_models

    async def test_upload_image_to_lmarena_success(self):
        image_data = b"fake_image_data"
        mime_type = "image/png"
        filename = "test.png"

        # Mock httpx response chain
        resp1 = MagicMock()
        resp1.raise_for_status.return_value = None
        resp1.text = '0:{"data":null}\n1:{"success":true,"data":{"uploadUrl":"http://upload","key":"img_key"}}'
        
        resp2 = MagicMock()
        resp2.raise_for_status.return_value = None
        
        resp3 = MagicMock()
        resp3.raise_for_status.return_value = None
        resp3.text = '0:{"data":null}\n1:{"success":true,"data":{"url":"http://download"}}'

        with patch('httpx.AsyncClient.post', side_effect=[resp1, resp3]) as mock_post, \
             patch('httpx.AsyncClient.put', return_value=resp2) as mock_put:
            
            result = await main.upload_image_to_lmarena(image_data, mime_type, filename)
            
            self.assertIsNotNone(result)
            self.assertEqual(result, ("img_key", "http://download"))

    async def test_upload_image_to_lmarena_fail_upload_url(self):
        resp1 = MagicMock()
        resp1.raise_for_status.return_value = None
        resp1.text = 'invalid_response'
        
        with patch('httpx.AsyncClient.post', return_value=resp1):
            result = await main.upload_image_to_lmarena(b"data", "image/png", "test.png")
            self.assertIsNone(result)

    async def test_process_message_content_with_base64_image(self):
        base64_img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAAAABJRU5ErkJggg=="
        content = [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": base64_img}}
        ]
        caps = {"inputCapabilities": {"image": True}}
        
        # Mock upload_image_to_lmarena directly to avoid httpx complexity
        with patch.object(main, 'upload_image_to_lmarena', new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = ("key1", "http://url1")
            
            text, attachments = await main.process_message_content(content, caps)
            
            self.assertEqual(text, "look at this")
            self.assertEqual(len(attachments), 1)
            self.assertEqual(attachments[0]["name"], "key1")

    async def test_rate_limit_api_key(self):
        main.api_key_usage = defaultdict(list)
        key_header = "Bearer test-key"
        
        # 1. First request
        key_data = await main.rate_limit_api_key(key_header)
        self.assertEqual(key_data["key"], "test-key")
        
        # 2. Saturate rate limit (60 max)
        current_time = time.time()
        main.api_key_usage["test-key"] = [current_time] * 60
        
        # 3. Next request should fail
        with self.assertRaises(HTTPException) as cm:
            await main.rate_limit_api_key(key_header)
        self.assertEqual(cm.exception.status_code, 429)

class TestChatCompletionsRetry(unittest.TestCase):
    def setUp(self):
        self.config = {
            "api_keys": [{"name":"k","key":"k","rpm":100}],
            "auth_tokens": ["t1", "t2"],
            "cf_clearance": "c",
            "password": "p",
            "auth_token": "t1",
            "usage_stats": {}
        }
        self.models = [{"id":"m1","publicName":"m1","organization":"o","capabilities":{"outputCapabilities":{"text":True}}}]
        
        # Manual patching
        self.original_get_config = main.get_config
        self.original_get_models = main.get_models
        self.original_refresh = main.refresh_recaptcha_token
        
        main.get_config = MagicMock(return_value=self.config)
        main.get_models = MagicMock(return_value=self.models)
        main.refresh_recaptcha_token = AsyncMock(return_value="recaptcha")
        
        # Patch heavy tasks
        self.patchers = [
            patch('main.get_initial_data', new_callable=AsyncMock),
            patch('main.periodic_refresh_task', new_callable=AsyncMock)
        ]
        for p in self.patchers:
            p.start()
            
        main.model_usage_stats = defaultdict(int)
        main.api_key_usage = defaultdict(list)
        main.chat_sessions = defaultdict(dict)
        main.current_token_index = 0
            
    def tearDown(self):
        main.get_config = self.original_get_config
        main.get_models = self.original_get_models
        main.refresh_recaptcha_token = self.original_refresh
        for p in self.patchers:
            p.stop()

    @patch('httpx.AsyncClient.post')
    def test_retry_on_429(self, mock_post):
        # 429 then 200
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.text = "Too Many Requests"
        
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.text = 'a0:"success"'
        resp_200.headers = {}
        resp_200.raise_for_status.return_value = None
        
        mock_post.side_effect = [resp_429, resp_200]
        
        with TestClient(main.app) as client:
            resp = client.post(
                "/api/v1/chat/completions",
                json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer k"}
            )
            
            # Should fail 200 if retry works
            self.assertEqual(resp.status_code, 200, f"Response: {resp.json() if resp.status_code != 200 else 'OK'}")
            self.assertEqual(mock_post.call_count, 2)

    @patch('httpx.AsyncClient.post')
    def test_retry_on_401(self, mock_post):
        # 401 then 200
        resp_401 = MagicMock()
        resp_401.status_code = 401
        resp_401.text = "Unauthorized"
        
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.text = 'a0:"success"'
        resp_200.headers = {}
        resp_200.raise_for_status.return_value = None
        
        mock_post.side_effect = [resp_401, resp_200]
        
        with TestClient(main.app) as client:
            resp = client.post(
                "/api/v1/chat/completions",
                json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer k"}
            )
            
            self.assertEqual(resp.status_code, 200, f"Response: {resp.json() if resp.status_code != 200 else 'OK'}")
            self.assertEqual(mock_post.call_count, 2)
            # Check logic: first token removed
            self.assertNotIn("t1", self.config["auth_tokens"])

            # Ensure we refreshed reCAPTCHA for both the initial send and the retry attempt.
            forced_calls = [
                call for call in main.refresh_recaptcha_token.await_args_list if call.kwargs.get("force") is True
            ]
            self.assertGreaterEqual(len(forced_calls), 2)

if __name__ == '__main__':
    unittest.main()
