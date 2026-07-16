import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import main


def _write_result(payload: dict) -> None:
    result = json.dumps(payload)
    result_path = str(os.environ.get("LM_UI_RESULT_PATH") or "").strip()
    if result_path:
        path = Path(result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result, encoding="utf-8")
    else:
        print("LM_UI_RESULT=" + result, flush=True)


async def _run() -> int:
    stage = "read_request"
    attempts_used = 0
    payload = {
        "response": None,
        "attempts": 0,
        "stage": stage,
        "error_code": "worker_exception",
    }
    exit_code = 1
    try:
        request = json.loads(sys.stdin.read() or "{}")
        timeout_seconds = int(request.get("timeout_seconds") or 220)
        max_attempts = max(2, int(os.environ.get("LM_UI_MAX_ATTEMPTS") or 2))
        attempt_timeout = max(20, min(100, timeout_seconds // max_attempts))
        response = None
        error_code = None

        stage = "initial_recovery"
        recovery = await main.ensure_authenticated_account(reason="ui_worker")
        browser_mode = recovery.browser_mode
        if not recovery.ok or not recovery.authenticated:
            payload = {
                "response": None,
                "attempts": 0,
                "stage": recovery.stage,
                "error_code": recovery.error_code
                or "authentication_recovery_failed",
                "browser_mode": recovery.browser_mode,
            }
            exit_code = 2
            return exit_code

        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            stage = "ui_request"
            try:
                response = await main._browser_ui_arena_response(
                    str(request.get("model") or ""),
                    str(request.get("prompt") or ""),
                    timeout_seconds=attempt_timeout,
                )
            except main.BrowserAuthRequired:
                error_code = "invalid_auth"
                stage = "auth_recovery"
                recovery = await main.ensure_authenticated_account(
                    reason="ui_login_required", force_recovery=True
                )
                browser_mode = recovery.browser_mode
                if not recovery.ok or not recovery.authenticated:
                    error_code = (
                        recovery.error_code or "authentication_recovery_failed"
                    )
                    stage = recovery.stage
                    break
                if attempt < max_attempts:
                    continue
            except main.BrowserChallengeUnresolved:
                error_code = "challenge_unresolved"
                if attempt < max_attempts:
                    continue
            if response:
                break
            print(
                f"Chrome UI worker attempt {attempt}/{max_attempts} returned no response; "
                "retrying with a fresh browser."
                if attempt < max_attempts
                else f"Chrome UI worker attempt {attempt}/{max_attempts} returned no response.",
                flush=True,
            )

        payload = {
            "response": response,
            "attempts": attempts_used,
            "stage": "complete" if response else stage,
            "error_code": None if response else error_code or "empty_ui_response",
            "browser_mode": browser_mode,
        }
        exit_code = 0 if response else 1
        return exit_code
    except Exception as exc:
        payload = {
            "response": None,
            "attempts": attempts_used,
            "stage": stage,
            "error_code": getattr(exc, "code", None) or "worker_exception",
            "error_type": type(exc).__name__,
            "error_detail": main.redact_text(str(exc))[:500],
        }
        exit_code = 1
        return exit_code
    finally:
        _write_result(payload)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
