import asyncio


def test_recaptcha_token_retries_when_execution_context_destroyed(monkeypatch):
    # Ensure local repo paths take precedence over any installed `src` package.
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import src.main as main

    async def fast_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(main.asyncio, "sleep", fast_sleep)

    class FakeMouse:
        async def move(self, *args, **kwargs):
            return None

        async def wheel(self, *args, **kwargs):
            return None

    class FakePage:
        def __init__(self):
            self.mouse = FakeMouse()
            self._token_result = None
            self._lib_checks = 0

        async def wait_for_load_state(self, *args, **kwargs):
            return None

        async def evaluate(self, script, *args, **kwargs):
            # First attempt to check grecaptcha fails due to navigation.
            if "!!(window.grecaptcha && window.grecaptcha.enterprise)" in script:
                self._lib_checks += 1
                if self._lib_checks == 1:
                    raise Exception(
                        "Page.evaluate: Execution context was destroyed, most likely because of a navigation."
                    )
                return True

            if "__token_result = 'PENDING'" in script:
                self._token_result = "PENDING"
                return None

            if "window.grecaptcha.enterprise.execute" in script:
                # Simulate async token becoming available.
                self._token_result = "fake-token"
                return None

            if script.strip().endswith("window.__token_result") or script.strip().endswith("mw:window.__token_result"):
                return self._token_result

            return None

    token = asyncio.run(main._get_recaptcha_v3_token_from_page(FakePage()))
    assert token == "fake-token"
