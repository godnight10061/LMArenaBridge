import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os
import json
import asyncio

# Add src to path so we can import main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fastapi.testclient import TestClient

# Mock the imports that might cause issues or are heavy
with patch.dict(sys.modules, {'camoufox.async_api': MagicMock(), 'camoufox': MagicMock()}):
    import main

class TestRecaptchaRetry(unittest.TestCase):
    def setUp(self):
        self._orig_models_json = None
        if os.path.exists("models.json"):
            with open("models.json", "r", encoding="utf-8") as f:
                self._orig_models_json = f.read()

        # Create temp config and models
        self.config = {
            "api_keys": [
                {
                    "name": "Test Key",
                    "key": "test-key",
                    "rpm": 1000,
                    "created": 1700000000
                }
            ],
            "auth_tokens": ["test-token"],
            "cf_clearance": "test-clearance",
            "password": "admin",
            "auth_token": "test-token",
            "usage_stats": {}
        }
        self.models = [
            {
                "id": "gpt-4",
                "publicName": "gpt-4",
                "organization": "openai",
                "capabilities": {
                    "outputCapabilities": {
                        "text": True
                    }
                }
            }
        ]
        
        with open("config.json", "w") as f:
            json.dump(self.config, f)
        with open("models.json", "w") as f:
            json.dump(self.models, f)

        # Manual patching of refresh_recaptcha_token
        self.original_refresh = main.refresh_recaptcha_token
        self.mock_refresh = AsyncMock(return_value="token_1")
        main.refresh_recaptcha_token = self.mock_refresh

        # Speed up app startup in tests (avoid 5s sleep + browser work)
        self.initial_data_patcher = patch.object(main, "get_initial_data", new=AsyncMock(return_value=None))
        self.periodic_refresh_patcher = patch.object(main, "periodic_refresh_task", new=AsyncMock(return_value=None))
        self.initial_data_patcher.start()
        self.periodic_refresh_patcher.start()
        
        # Patch httpx
        self.post_patcher = patch('httpx.AsyncClient.post', new_callable=AsyncMock)
        self.mock_post = self.post_patcher.start()
        
    def tearDown(self):
        main.refresh_recaptcha_token = self.original_refresh
        self.initial_data_patcher.stop()
        self.periodic_refresh_patcher.stop()
        self.post_patcher.stop()
        
        # Clean up files
        if os.path.exists("config.json"):
            os.remove("config.json")
        if self._orig_models_json is not None:
            with open("models.json", "w", encoding="utf-8") as f:
                f.write(self._orig_models_json)
        elif os.path.exists("models.json"):
            os.remove("models.json")

    def test_recaptcha_failure_retry(self):
        """
        Test that if the API returns "recaptcha validation failed",
        the system refreshes the token and retries.
        """
        # Setup mocks
        self.mock_refresh.side_effect = ["token_1", "token_2", "token_3"]
        
        # Mock responses
        from httpx import HTTPStatusError, Request, Response
        request = Request("POST", "https://lmarena.ai")
        
        # Response 1: Failure
        response_fail_obj = Response(400, request=request, text='{"error": "recaptcha validation failed"}')
        error_fail = HTTPStatusError("400 Bad Request", request=request, response=response_fail_obj)
        
        response_fail = MagicMock()
        response_fail.status_code = 400
        response_fail.text = '{"error": "recaptcha validation failed"}'
        response_fail.json.return_value = {"error": "recaptcha validation failed"}
        response_fail.raise_for_status.side_effect = error_fail

        # Response 2: Success
        response_success = MagicMock()
        response_success.status_code = 200
        response_success.text = 'a0:"Hello world"'
        response_success.headers = {}
        response_success.raise_for_status.return_value = None

        self.mock_post.side_effect = [response_fail, response_success]

        # Use context manager
        with TestClient(main.app) as client:
            # Make the request
            response = client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}]
                },
                headers={"Authorization": "Bearer test-key"}
            )

            print(f"Test Status Code: {response.status_code}")
            if response.status_code != 200:
                print(f"Error Detail: {response.json()}")
            
            self.assertEqual(response.status_code, 200)
            self.assertIn("Hello world", response.json()['choices'][0]['message']['content'])
            
            # Verify retry
            self.assertEqual(self.mock_post.call_count, 2, "Should have retried request")

if __name__ == '__main__':
    unittest.main()
