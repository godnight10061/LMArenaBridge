import httpx
from fastapi.testclient import TestClient


def test_issue_27_recaptcha_validation_failed_triggers_retry(monkeypatch):
    # Ensure local repo paths take precedence over any installed `src` package.
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import src.main as main

    # Avoid running real startup tasks (network/browser + file writes) in unit tests.
    original_startup_handlers = list(main.app.router.on_startup)
    original_shutdown_handlers = list(main.app.router.on_shutdown)
    main.app.router.on_startup.clear()
    main.app.router.on_shutdown.clear()

    main.app.dependency_overrides[main.rate_limit_api_key] = lambda: {"key": "test-api-key", "rpm": 60}

    monkeypatch.setattr(
        main,
        "get_models",
        lambda: [
            {
                "publicName": "test-model",
                "id": "test-model-id",
                "organization": "test-org",
                "capabilities": {
                    "inputCapabilities": {"text": True},
                    "outputCapabilities": {"text": True},
                },
            }
        ],
    )

    monkeypatch.setattr(main, "save_config", lambda config: None)
    monkeypatch.setattr(main, "get_request_headers", lambda: {})
    monkeypatch.setattr(main, "get_request_headers_with_token", lambda token: {})
    monkeypatch.setattr(main, "get_next_auth_token", lambda exclude_tokens=None: "arena-auth-prod-v1-token")

    issued_tokens = ["token-1", "token-2"]

    async def fake_refresh_recaptcha_token(force: bool = False):
        return issued_tokens.pop(0)

    monkeypatch.setattr(main, "refresh_recaptcha_token", fake_refresh_recaptcha_token)

    request_count = {"n": 0}

    async def fake_post(self, url, json=None, headers=None, timeout=None):
        request_count["n"] += 1
        request = httpx.Request("POST", url)

        if request_count["n"] == 1:
            return httpx.Response(
                400,
                json={"error": "recaptcha validation failed"},
                request=request,
            )

        return httpx.Response(
            200,
            text='a0:"Hello"\nad:{"finishReason":"stop"}\n',
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    client = TestClient(main.app)
    response = client.post(
        "/api/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    data = response.json()

    assert "error" not in data
    assert data["choices"][0]["message"]["content"] == "Hello"
    assert request_count["n"] == 2

    main.app.router.on_startup[:] = original_startup_handlers
    main.app.router.on_shutdown[:] = original_shutdown_handlers
    main.app.dependency_overrides.clear()


def test_issue_27_recaptcha_validation_failed_triggers_v2_fallback(monkeypatch):
    # Ensure local repo paths take precedence over any installed `src` package.
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import src.main as main

    # Avoid running real startup tasks (network/browser + file writes) in unit tests.
    original_startup_handlers = list(main.app.router.on_startup)
    original_shutdown_handlers = list(main.app.router.on_shutdown)
    main.app.router.on_startup.clear()
    main.app.router.on_shutdown.clear()

    # Enable v2 fallback in code under test.
    monkeypatch.setenv("LMABRIDGE_RECAPTCHA_V2_MODE", "auto")

    main.app.dependency_overrides[main.rate_limit_api_key] = lambda: {"key": "test-api-key", "rpm": 60}

    monkeypatch.setattr(
        main,
        "get_models",
        lambda: [
            {
                "publicName": "test-model",
                "id": "test-model-id",
                "organization": "test-org",
                "capabilities": {
                    "inputCapabilities": {"text": True},
                    "outputCapabilities": {"text": True},
                },
            }
        ],
    )

    monkeypatch.setattr(main, "save_config", lambda config: None)
    monkeypatch.setattr(main, "get_request_headers", lambda: {})
    monkeypatch.setattr(main, "get_request_headers_with_token", lambda token: {})
    monkeypatch.setattr(main, "get_next_auth_token", lambda exclude_tokens=None: "arena-auth-prod-v1-token")

    async def fake_refresh_recaptcha_token(force: bool = False):
        return "v3-token"

    monkeypatch.setattr(main, "refresh_recaptcha_token", fake_refresh_recaptcha_token)

    v2_calls = []

    async def fake_refresh_recaptcha_v2_token(force: bool = False):
        v2_calls.append(force)
        return "v2-token"

    monkeypatch.setattr(main, "refresh_recaptcha_v2_token", fake_refresh_recaptcha_v2_token)

    request_count = {"n": 0}
    sent_payloads = []

    async def fake_post(self, url, json=None, headers=None, timeout=None):
        import json as jsonlib

        request_count["n"] += 1
        sent_payloads.append(jsonlib.loads(jsonlib.dumps(json)))
        request = httpx.Request("POST", url)

        if request_count["n"] == 1:
            return httpx.Response(
                403,
                json={"error": "recaptcha validation failed"},
                request=request,
            )

        return httpx.Response(
            200,
            text='a0:"Hello"\nad:{"finishReason":"stop"}\n',
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    client = TestClient(main.app)
    response = client.post(
        "/api/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    data = response.json()

    assert "error" not in data
    assert data["choices"][0]["message"]["content"] == "Hello"
    assert request_count["n"] == 2

    assert v2_calls == [True]
    assert "recaptchaV2Token" in sent_payloads[1]
    assert sent_payloads[1]["recaptchaV2Token"] == "v2-token"
    assert "recaptchaV3Token" not in sent_payloads[1]

    main.app.router.on_startup[:] = original_startup_handlers
    main.app.router.on_shutdown[:] = original_shutdown_handlers
    main.app.dependency_overrides.clear()
