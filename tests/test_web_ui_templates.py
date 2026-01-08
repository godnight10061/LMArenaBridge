import unittest


class TestWebUiTemplates(unittest.TestCase):
    def test_login_template_renders(self):
        from src import web_ui

        html = web_ui.render_login_page(error=False)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Login - LMArena Bridge", html)

    def test_dashboard_template_renders(self):
        from src import web_ui

        config = {
            "api_keys": [],
            "auth_token": "",
            "auth_tokens": [],
            "cf_clearance": "",
        }

        html = web_ui.render_dashboard_page(
            config=config,
            text_models=[],
            model_usage_stats={},
            token_status="Not Set",
            token_class="status-bad",
            cf_status="Not Set",
            cf_class="status-bad",
            keys_html="",
            stats_html="",
            models_html="",
        )

        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("LMArena Bridge Dashboard", html)


if __name__ == "__main__":
    unittest.main()

