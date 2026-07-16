import unittest

from src.transport import _normalize_userscript_proxy_url


class TestUserscriptProxyUrlNormalization(unittest.TestCase):
    def test_normalizes_arena_urls_and_preserves_external_urls(self) -> None:
        self.assertEqual(
            _normalize_userscript_proxy_url("https://arena.ai/nextjs-api/stream/create-evaluation"),
            "/nextjs-api/stream/create-evaluation",
        )
        self.assertEqual(
            _normalize_userscript_proxy_url("https://arena.ai/nextjs-api/sign-up?x=1"),
            "/nextjs-api/sign-up?x=1",
        )
        self.assertEqual(
            _normalize_userscript_proxy_url("/nextjs-api/stream/create-evaluation"),
            "/nextjs-api/stream/create-evaluation",
        )
        self.assertEqual(
            _normalize_userscript_proxy_url("https://example.com/foo"),
            "https://example.com/foo",
        )


if __name__ == "__main__":
    unittest.main()
