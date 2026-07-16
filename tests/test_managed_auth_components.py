import base64
import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from account_recovery import (
    classify_failure,
    inspect_auth_cookie,
    merge_authenticated_tokens,
    redact_text,
)
from temp_mail import extract_verification_url


def _cookie(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return "base64-" + encoded.rstrip("=")


class TestManagedAuthRealComponents(unittest.TestCase):
    def test_authenticated_cookie_is_accepted(self):
        result = inspect_auth_cookie(
            _cookie(
                {
                    "expires_at": int(time.time()) + 3600,
                    "user": {
                        "aud": "authenticated",
                        "role": "authenticated",
                        "email": "bridge@example.invalid",
                        "is_anonymous": False,
                    },
                }
            )
        )
        self.assertTrue(result.authenticated)
        self.assertFalse(result.anonymous)

    def test_anonymous_cookie_is_rejected(self):
        result = inspect_auth_cookie(
            _cookie(
                {
                    "expires_at": int(time.time()) + 3600,
                    "user": {
                        "aud": "authenticated",
                        "role": "authenticated",
                        "is_anonymous": True,
                    },
                }
            )
        )
        self.assertFalse(result.authenticated)
        self.assertTrue(result.anonymous)
        self.assertEqual(result.reason, "anonymous_auth")

    def test_expired_cookie_is_rejected(self):
        result = inspect_auth_cookie(
            _cookie(
                {
                    "expires_at": int(time.time()) - 5,
                    "user": {
                        "aud": "authenticated",
                        "role": "authenticated",
                        "is_anonymous": False,
                    },
                }
            )
        )
        self.assertFalse(result.authenticated)
        self.assertTrue(result.expired)

    def test_verification_link_parser_prefers_arena_callback(self):
        url = extract_verification_url(
            "Open https://arena.ai/nextjs-api/callback/email?token=secret-value to continue."
        )
        self.assertTrue(url.startswith("https://arena.ai/nextjs-api/callback/email"))

    def test_redaction_removes_credentials(self):
        redacted = redact_text(
            "email=user@example.com password=hunter2 "
            "token=https://arena.ai/callback?token=abc base64-abcdef"
        )
        self.assertNotIn("user@example.com", redacted)
        self.assertNotIn("hunter2", redacted)
        self.assertNotIn("token=abc", redacted)
        self.assertNotIn("base64-abcdef", redacted)

    def test_account_creation_is_not_triggered_for_non_auth_failures(self):
        self.assertEqual(classify_failure(429, "rate limit"), "rate_limited")
        self.assertEqual(classify_failure(404, "model not found"), "model_missing")
        self.assertEqual(
            classify_failure(400, "recaptcha validation failed"),
            "challenge_retryable",
        )
        self.assertEqual(classify_failure(401, "expired"), "invalid_auth")

    def test_authenticated_merge_drops_anonymous_and_expired_tokens(self):
        current = _cookie(
            {
                "expires_at": int(time.time()) + 3600,
                "user": {"role": "authenticated", "is_anonymous": False},
            }
        )
        anonymous = _cookie(
            {
                "expires_at": int(time.time()) + 3600,
                "user": {"role": "authenticated", "is_anonymous": True},
            }
        )
        expired = _cookie(
            {
                "expires_at": int(time.time()) - 10,
                "user": {"role": "authenticated", "is_anonymous": False},
            }
        )
        merged = merge_authenticated_tokens(
            current, [anonymous, expired, "legacy-opaque-token", current]
        )
        self.assertEqual(merged, [current, "legacy-opaque-token"])


if __name__ == "__main__":
    unittest.main()
