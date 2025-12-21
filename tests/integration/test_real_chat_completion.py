import os
import json

import pytest
from fastapi.testclient import TestClient


def _run_real_tests() -> bool:
    return os.environ.get("RUN_REAL_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}


def _reproduce_issue_27() -> bool:
    return os.environ.get("REPRODUCE_ISSUE_27", "").strip().lower() in {"1", "true", "yes", "on"}


@pytest.mark.skipif(
    not _run_real_tests() or not _reproduce_issue_27(),
    reason="Set RUN_REAL_TESTS=1 and REPRODUCE_ISSUE_27=1 to run real upstream repro test",
)
def test_real_reproduce_issue_27_when_v2_off(monkeypatch):
    """
    Real repro for issue #27 (upstream rejects reCAPTCHA v3):
    - Force `LMABRIDGE_RECAPTCHA_V2_MODE=off`
    - Expect an upstream failure containing "recaptcha ... validation failed"

    This is intentionally optional because some environments may still accept v3.
    """
    # Ensure local repo paths take precedence over any installed `src` package.
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    import src.main as main

    # Keep logs on for real runs (payload/token logs are redacted).
    main.DEBUG = True

    monkeypatch.setenv("LMABRIDGE_RECAPTCHA_V2_MODE", "off")

    main.app.router.on_startup.clear()
    main.app.router.on_shutdown.clear()
    main.app.dependency_overrides[main.rate_limit_api_key] = lambda: {"key": "real-test", "rpm": 999}

    models = main.get_models()
    if not models:
        pytest.skip("No models available. Run the server once to generate models.json.")

    preferred = "gemini-3-pro-grounding"
    selected = next((m for m in models if m.get("publicName") == preferred), None) or models[0]
    model_name = selected.get("publicName") or preferred

    client = TestClient(main.app)
    response = client.post(
        "/api/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": "Say 'ok'."}],
            "stream": False,
        },
    )

    data = response.json()
    if "error" not in data:
        pytest.skip("Upstream accepted reCAPTCHA v3; could not reproduce issue #27 here.")

    message = (data.get("error") or {}).get("message", "")
    lowered = message.lower()
    assert (
        ("recaptcha" in lowered and "validation failed" in lowered)
        or "max retries exceeded" in lowered
    ), message


@pytest.mark.skipif(not _run_real_tests(), reason="Set RUN_REAL_TESTS=1 to run real browser + upstream tests")
def test_real_chat_completion_default_v2_mode(monkeypatch):
    """
    Repro for issue #27 in a REAL environment:
    - Without explicitly enabling reCAPTCHA v2 mode, LMArena often rejects v3 tokens
      with `{"error":"recaptcha validation failed"}` resulting in upstream 403s.

    Expected behavior (after fix): on desktop/Windows, the bridge should default to an
    interactive v2 fallback when v3 is rejected, allowing a successful completion.
    """
    # Ensure local repo paths take precedence over any installed `src` package.
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    import src.main as main

    # Keep logs on for real runs (payload/token logs are redacted).
    main.DEBUG = True

    # Ensure we're testing the default behavior (no env override).
    monkeypatch.delenv("LMABRIDGE_RECAPTCHA_V2_MODE", raising=False)

    # Avoid long startup tasks (periodic refresh, model downloads, etc.) in tests.
    main.app.router.on_startup.clear()
    main.app.router.on_shutdown.clear()

    # Skip API-key auth/rate limiting in tests.
    main.app.dependency_overrides[main.rate_limit_api_key] = lambda: {"key": "real-test", "rpm": 999}

    models = main.get_models()
    if not models:
        pytest.skip("No models available. Run the server once to generate models.json.")

    preferred = "gemini-3-pro-grounding"
    selected = next((m for m in models if m.get("publicName") == preferred), None) or models[0]
    model_name = selected.get("publicName") or preferred

    print(
        "\n[real test] If a reCAPTCHA window opens, solve it to allow the request to complete.\n"
        "[real test] If you want to force non-interactive behavior: set LMABRIDGE_RECAPTCHA_V2_MODE=off.\n"
    )

    client = TestClient(main.app)
    response = client.post(
        "/api/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": "Say 'ok'."}],
            "stream": False,
        },
    )

    data = response.json()
    assert "error" not in data, data.get("error") or data
    assert data["choices"][0]["message"]["content"].strip()


@pytest.mark.skipif(not _run_real_tests(), reason="Set RUN_REAL_TESTS=1 to run real browser + upstream tests")
def test_real_chat_completion_stream_default_v2_mode(monkeypatch):
    """
    Same as `test_real_chat_completion_default_v2_mode`, but exercises the STREAMING path
    to reproduce/fix the reported `LMArena API error: 403` stream failure.
    """
    # Ensure local repo paths take precedence over any installed `src` package.
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    import src.main as main

    # Keep logs on for real runs (payload/token logs are redacted).
    main.DEBUG = True

    # Ensure we're testing the default behavior (no env override).
    monkeypatch.delenv("LMABRIDGE_RECAPTCHA_V2_MODE", raising=False)

    # Avoid long startup tasks (periodic refresh, model downloads, etc.) in tests.
    main.app.router.on_startup.clear()
    main.app.router.on_shutdown.clear()

    # Skip API-key auth/rate limiting in tests.
    main.app.dependency_overrides[main.rate_limit_api_key] = lambda: {"key": "real-test", "rpm": 999}

    models = main.get_models()
    if not models:
        pytest.skip("No models available. Run the server once to generate models.json.")

    preferred = "gemini-3-pro-grounding"
    selected = next((m for m in models if m.get("publicName") == preferred), None) or models[0]
    model_name = selected.get("publicName") or preferred

    print(
        "\n[real test] STREAMING request.\n"
        "[real test] If a reCAPTCHA window opens, solve it to allow the request to complete.\n"
        "[real test] If you want to force non-interactive behavior: set LMABRIDGE_RECAPTCHA_V2_MODE=off.\n"
    )

    client = TestClient(main.app)
    with client.stream(
        "POST",
        "/api/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": "Say 'ok'."}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200

        collected = ""
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = (
                raw_line.decode("utf-8", errors="replace")
                if isinstance(raw_line, (bytes, bytearray))
                else raw_line
            )
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if "error" in obj:
                raise AssertionError(obj["error"])
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    collected += piece

        assert collected.strip()
