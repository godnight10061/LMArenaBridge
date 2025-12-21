import asyncio
import os

import pytest


def _run_real_tests() -> bool:
    return os.environ.get("RUN_REAL_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}


@pytest.mark.skipif(not _run_real_tests(), reason="Set RUN_REAL_TESTS=1 to run real browser tests")
def test_get_recaptcha_v3_token_real():
    import src.main as main

    token = asyncio.run(asyncio.wait_for(main.get_recaptcha_v3_token(), timeout=120))
    assert isinstance(token, str)
    assert token

