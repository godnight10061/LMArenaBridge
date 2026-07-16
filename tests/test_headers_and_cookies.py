import json
import os
import unittest

from src import main


def _parse_cookie_header(cookie_header: str) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for part in (cookie_header or "").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        if not name:
            continue
        cookies[name] = value
    return cookies


class TestHeadersAndCookies(unittest.TestCase):
    def setUp(self):
        self._orig_config_json = None
        if os.path.exists("config.json"):
            with open("config.json", "r", encoding="utf-8") as f:
                self._orig_config_json = f.read()

    def tearDown(self):
        if os.path.exists("config.json"):
            os.remove("config.json")
        if self._orig_config_json is not None:
            with open("config.json", "w", encoding="utf-8") as f:
                f.write(self._orig_config_json)

    def test_build_cookie_header_includes_clearance_and_auth(self):
        config = {
            "password": "admin",
            "cf_clearance": "test-clearance",
            "auth_tokens": ["t1"],
            "auth_token": "t1",
            "cookie_jar": [],
            "api_keys": [],
            "usage_stats": {},
        }
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        header = main.build_lmarena_cookie_header("token123")
        cookies = _parse_cookie_header(header)
        self.assertEqual(cookies.get("cf_clearance"), "test-clearance")
        self.assertEqual(cookies.get("arena-auth-prod-v1"), "token123")

    def test_cookie_jar_is_included_and_overridden_by_config(self):
        config = {
            "password": "admin",
            "cf_clearance": "clearance-from-config",
            "auth_tokens": ["t1"],
            "auth_token": "t1",
            "cookie_jar": [
                {"name": "__cf_bm", "value": "bm123"},
                {"name": "cf_clearance", "value": "stale-clearance"},
                {"name": "arena-auth-prod-v1", "value": "stale-token"},
            ],
            "api_keys": [],
            "usage_stats": {},
        }
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        header = main.build_lmarena_cookie_header("fresh-token")
        cookies = _parse_cookie_header(header)

        self.assertEqual(cookies.get("__cf_bm"), "bm123")
        self.assertEqual(cookies.get("cf_clearance"), "clearance-from-config")
        self.assertEqual(cookies.get("arena-auth-prod-v1"), "fresh-token")

    def test_get_request_headers_with_token_uses_cookie_builder(self):
        config = {
            "password": "admin",
            "cf_clearance": "test-clearance",
            "auth_tokens": ["t1"],
            "auth_token": "t1",
            "cookie_jar": [{"name": "__cf_bm", "value": "bm123"}],
            "api_keys": [],
            "usage_stats": {},
        }
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        headers = main.get_request_headers_with_token("token123", "recaptcha123")
        self.assertEqual(headers.get("Referer"), "https://arena.ai/?mode=direct")
        self.assertIn("Sec-Fetch-Site", headers)
        self.assertEqual(headers.get("X-Recaptcha-Token"), "recaptcha123")
        self.assertEqual(headers.get("X-Recaptcha-Action"), main.RECAPTCHA_ACTION)

        cookies = _parse_cookie_header(headers.get("Cookie", ""))
        self.assertEqual(cookies.get("__cf_bm"), "bm123")
        self.assertEqual(cookies.get("cf_clearance"), "test-clearance")
        self.assertEqual(cookies.get("arena-auth-prod-v1"), "token123")

    def test_foreign_and_oversized_cookies_are_excluded(self):
        config = {
            "password": "admin",
            "cf_clearance": "test-clearance",
            "auth_tokens": ["t1"],
            "auth_token": "t1",
            "cookie_jar": [
                {"name": "euconsent", "value": "x" * 20000, "domain": "example.com"},
                {"name": "__cf_bm", "value": "bm123", "domain": ".arena.ai"},
                {"name": "cf_clearance", "value": "wrong-domain", "domain": ".canva.com"},
            ],
            "api_keys": [],
            "usage_stats": {},
        }
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        header = main.build_lmarena_cookie_header("token123")
        cookies = _parse_cookie_header(header)

        self.assertNotIn("euconsent", cookies)
        self.assertEqual(cookies.get("__cf_bm"), "bm123")
        self.assertEqual(cookies.get("cf_clearance"), "test-clearance")
        self.assertLess(len(header), 8192)

    def test_cookie_persistence_keeps_only_arena_session_cookies(self):
        simplified = main._simplify_cookie_jar([
            {"name": "cf_clearance", "value": "arena", "domain": ".arena.ai"},
            {"name": "cf_clearance", "value": "foreign", "domain": ".canva.com"},
            {"name": "tracking", "value": "value", "domain": ".arena.ai"},
        ])

        self.assertEqual(simplified, [
            {"name": "cf_clearance", "value": "arena", "domain": ".arena.ai"},
        ])

    def test_split_arena_auth_cookies_are_combined(self):
        cookies = [
            {"name": "arena-auth-prod-v1.0", "value": "base64-first", "domain": "arena.ai"},
            {"name": "arena-auth-prod-v1.1", "value": "-second", "domain": "arena.ai"},
        ]

        self.assertEqual(
            main._combine_split_arena_auth_cookies(cookies),
            "base64-first-second",
        )
        simplified = main._simplify_cookie_jar(cookies)
        self.assertEqual(simplified[0]["name"], "arena-auth-prod-v1")
        self.assertEqual(simplified[0]["value"], "base64-first-second")


if __name__ == "__main__":
    unittest.main()
