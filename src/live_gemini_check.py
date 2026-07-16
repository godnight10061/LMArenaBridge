"""Real end-to-end Gemini check through the local OpenAI-compatible bridge."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

try:
    import account_recovery as _account_recovery
except ModuleNotFoundError:
    from . import account_recovery as _account_recovery

redact_text = _account_recovery.redact_text


class LiveCheckFailure(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


_PREFERRED_MODEL_CANDIDATES = [
    "gpt-5.2-chat-latest",
    "claude-sonnet-4-6",
    "grok-4.3",
    "qwen3.7-plus",
    "deepseek-v4-flash",
    "gemini-3.5-flash-high",
    "gpt-5.5-instant",
    "qwen3.5-flash",
]

_MODEL_NAME_EXCLUSIONS = (
    "search",
    "image",
    "video",
    "audio",
    "omni",
    "codex",
    "thinking",
    "reasoning",
    "no-system",
    "dlp-test",
)


def _bounded_request_delay(value: float) -> float:
    """Keep live request pacing bounded without weakening the matrix."""
    return max(0.0, min(float(value), 30.0))


def _pause_between_requests(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _load_api_key() -> str:
    configured = str(os.environ.get("LM_BRIDGE_API_KEY") or "").strip()
    if configured:
        return configured
    config_path = Path(os.environ.get("LM_CONFIG_FILE") or "config.json")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        payload = {}
    keys = payload.get("api_keys") or []
    if keys and isinstance(keys[0], dict):
        return str(keys[0].get("key") or "")
    raise LiveCheckFailure("configuration", "No bridge API key is available")


def _wait_for_health(client: httpx.Client, health_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            response = client.get(health_url)
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}"
        except (httpx.HTTPError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise LiveCheckFailure("health", f"Bridge health wait timed out: {last_error}")


def _extract_message(payload: dict[str, Any]) -> str:
    try:
        return str(payload["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        return ""


def _non_streaming(
    client: httpx.Client, url: str, headers: dict[str, str], model: str, nonce: str
) -> dict[str, Any]:
    prompt = f"Reply with exactly {nonce} and no other text."
    response = client.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
    )
    if response.status_code != 200:
        raise LiveCheckFailure(
            "non_streaming_request",
            f"Bridge returned HTTP {response.status_code} ({len(response.text)} chars)",
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise LiveCheckFailure("non_streaming_parse", "Bridge returned invalid JSON") from exc
    content = _extract_message(payload)
    if not content.strip():
        raise LiveCheckFailure("non_streaming_content", "Assistant response was empty")
    if nonce not in content:
        raise LiveCheckFailure(
            "non_streaming_nonce",
            f"Assistant response did not contain the requested nonce ({len(content)} chars)",
        )
    return {
        "status": "passed",
        "response_chars": len(content),
        "finish_reason": (payload.get("choices") or [{}])[0].get("finish_reason"),
    }


def _streaming(
    client: httpx.Client, url: str, headers: dict[str, str], model: str, nonce: str
) -> dict[str, Any]:
    prompt = f"Reply with exactly {nonce} and no other text."
    content_parts: list[str] = []
    saw_terminal = False
    saw_done = False
    with client.stream(
        "POST",
        url,
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        },
    ) as response:
        if response.status_code != 200:
            raise LiveCheckFailure(
                "streaming_request",
                f"Bridge returned HTTP {response.status_code}",
            )
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                continue
            try:
                payload = json.loads(data)
            except ValueError as exc:
                raise LiveCheckFailure(
                    "streaming_parse", "Streaming response contained invalid JSON"
                ) from exc
            choices = payload.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(str(delta["content"]))
            if choice.get("finish_reason"):
                saw_terminal = True
    content = "".join(content_parts)
    if nonce not in content:
        raise LiveCheckFailure(
            "streaming_nonce",
            f"Streaming response did not contain the requested nonce ({len(content)} chars)",
        )
    if not saw_terminal:
        raise LiveCheckFailure("streaming_terminal", "No terminal streaming chunk was received")
    if not saw_done:
        raise LiveCheckFailure("streaming_done", "Streaming response omitted [DONE]")
    return {"status": "passed", "response_chars": len(content), "done": True}


def _contextual(
    client: httpx.Client, url: str, headers: dict[str, str], model: str, nonce: str
) -> dict[str, Any]:
    """Verify that the Gemini UI path receives ordered prior user/assistant/tool context."""
    response = client.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Use the supplied conversation and answer the final user turn exactly.",
                },
                {"role": "user", "content": "Ask the lookup tool for the saved marker."},
                {
                    "role": "assistant",
                    "content": "I will use the lookup tool.",
                    "reasoning_content": "This provider-only field must not be replayed.",
                    "tool_calls": [
                        {
                            "id": "live-context-call",
                            "type": "function",
                            "function": {
                                "name": "lookup_marker",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "live-context-call",
                    "content": f"saved_marker={nonce}",
                },
                {
                    "role": "user",
                    "content": "What exact saved marker did the tool return? Reply with only the marker, no explanation.",
                },
            ],
            "stream": False,
        },
    )
    if response.status_code != 200:
        raise LiveCheckFailure(
            "contextual_request",
            f"Bridge returned HTTP {response.status_code} ({len(response.text)} chars)",
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise LiveCheckFailure("contextual_parse", "Bridge returned invalid JSON") from exc
    content = _extract_message(payload)
    if nonce not in content:
        raise LiveCheckFailure(
            "contextual_nonce",
            f"Contextual response omitted the earlier tool marker ({len(content)} chars)",
        )
    return {
        "status": "passed",
        "response_chars": len(content),
        "finish_reason": (payload.get("choices") or [{}])[0].get("finish_reason"),
    }


def _select_model_candidates(
    records: list[dict[str, Any]], *, primary_model: str, limit: int
) -> list[dict[str, str]]:
    available = {
        str(record.get("id") or ""): {
            "model": str(record.get("id") or ""),
            "provider": str(record.get("owned_by") or "lmarena"),
        }
        for record in records
        if str(record.get("id") or "")
    }
    selected: list[dict[str, str]] = []
    seen_models = {primary_model}

    def add(name: str) -> None:
        if len(selected) >= limit or name in seen_models or name not in available:
            return
        selected.append(available[name])
        seen_models.add(name)

    for name in _PREFERRED_MODEL_CANDIDATES:
        add(name)

    seen_providers = {item["provider"].lower() for item in selected}
    fallback = sorted(available.values(), key=lambda item: (item["provider"], item["model"]))
    for item in fallback:
        lowered = item["model"].lower()
        if any(marker in lowered for marker in _MODEL_NAME_EXCLUSIONS):
            continue
        provider = item["provider"].lower()
        if provider not in seen_providers:
            add(item["model"])
            seen_providers.add(provider)
    for item in fallback:
        lowered = item["model"].lower()
        if not any(marker in lowered for marker in _MODEL_NAME_EXCLUSIONS):
            add(item["model"])
    return selected[:limit]


def _model_smoke(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    candidate: dict[str, str],
    *,
    timeout: float,
) -> dict[str, Any]:
    model = candidate["model"]
    nonce = "BRIDGE_MODEL_OK_" + uuid.uuid4().hex[:12].upper()
    started = time.monotonic()
    result: dict[str, Any] = {
        "model": model,
        "provider": candidate["provider"],
        "status": "failed",
    }
    try:
        response = client.post(
            url,
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": f"Reply with exactly {nonce} and no other text."}
                ],
                "stream": False,
            },
            timeout=timeout,
        )
        result["http_status"] = response.status_code
        if response.status_code != 200:
            result["error"] = f"http_{response.status_code}"
            return result
        payload = response.json()
        content = _extract_message(payload)
        result["response_chars"] = len(content)
        if nonce not in content:
            result["error"] = "nonce_missing"
            return result
        result["status"] = "passed"
        return result
    except (httpx.HTTPError, ValueError) as exc:
        result["error"] = type(exc).__name__
        return result
    finally:
        result["latency_seconds"] = round(time.monotonic() - started, 3)


def _successful_candidate_models(attempts: list[dict[str, Any]], *, primary_model: str) -> set[str]:
    """Return distinct passing smoke-test models, excluding mandatory Gemini."""
    return {
        str(attempt.get("model") or "")
        for attempt in attempts
        if attempt.get("status") == "passed"
        and str(attempt.get("model") or "")
        and str(attempt.get("model") or "") != primary_model
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--health-timeout", type=float, default=420)
    parser.add_argument("--request-timeout", type=float, default=360)
    parser.add_argument("--minimum-model-successes", type=int, default=5)
    parser.add_argument("--model-candidate-limit", type=int, default=8)
    parser.add_argument("--model-smoke-timeout", type=float, default=180)
    parser.add_argument(
        "--request-delay",
        type=float,
        default=float(os.environ.get("LM_LIVE_REQUEST_DELAY_SECONDS") or 5),
        help="Seconds to wait between real Arena requests (bounded to 0-30).",
    )
    args = parser.parse_args()
    request_delay = _bounded_request_delay(args.request_delay)

    base_url = args.base_url.rstrip("/")
    artifact_dir = Path(os.environ.get("LM_LIVE_ARTIFACT_DIR") or ".runtime/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "live-gemini-summary.json"
    summary: dict[str, Any] = {
        "model": args.model,
        "started_at": int(time.time()),
        "status": "failed",
        "request_delay_seconds": request_delay,
    }
    try:
        api_key = _load_api_key()
        headers = {"Authorization": f"Bearer {api_key}"}
        timeout = httpx.Timeout(args.request_timeout, connect=30)
        with httpx.Client(timeout=timeout) as client:
            summary["health"] = _wait_for_health(client, f"{base_url}/health", args.health_timeout)
            if not bool((summary["health"].get("checks") or {}).get("model_catalog_fresh")):
                raise LiveCheckFailure(
                    "model_refresh", "Bridge started without a fresh Arena model catalog"
                )
            models_response = client.get(f"{base_url}/models", headers=headers)
            if models_response.status_code != 200:
                raise LiveCheckFailure(
                    "models", f"Models endpoint returned HTTP {models_response.status_code}"
                )
            model_records = models_response.json().get("data") or []
            models = [item.get("id") for item in model_records]
            if args.model not in models:
                raise LiveCheckFailure(
                    "models", f"Required model {args.model} was not exposed by the bridge"
                )

            nonce_a = "BRIDGE_OK_" + uuid.uuid4().hex[:12].upper()
            nonce_b = "BRIDGE_STREAM_OK_" + uuid.uuid4().hex[:12].upper()
            completion_url = f"{base_url}/chat/completions"
            summary["non_streaming"] = _non_streaming(
                client, completion_url, headers, args.model, nonce_a
            )
            _pause_between_requests(request_delay)
            summary["streaming"] = _streaming(client, completion_url, headers, args.model, nonce_b)
            _pause_between_requests(request_delay)
            nonce_c = "BRIDGE_CONTEXT_OK_" + uuid.uuid4().hex[:12].upper()
            summary["contextual"] = _contextual(
                client, completion_url, headers, args.model, nonce_c
            )
            matrix = [
                {
                    "model": args.model,
                    "provider": next(
                        (
                            str(item.get("owned_by") or "lmarena")
                            for item in model_records
                            if item.get("id") == args.model
                        ),
                        "google",
                    ),
                    "status": "passed",
                    "mandatory": True,
                }
            ]
            candidates = _select_model_candidates(
                model_records,
                primary_model=args.model,
                limit=max(0, args.model_candidate_limit),
            )
            for candidate in candidates:
                _pause_between_requests(request_delay)
                result = _model_smoke(
                    client,
                    completion_url,
                    headers,
                    candidate,
                    timeout=args.model_smoke_timeout,
                )
                matrix.append(result)
                successful_candidates = _successful_candidate_models(
                    matrix, primary_model=args.model
                )
                if len(successful_candidates) >= args.minimum_model_successes:
                    break
            successful_candidates = _successful_candidate_models(matrix, primary_model=args.model)
            summary["model_matrix"] = {
                "required_successes": args.minimum_model_successes,
                "successful_models": len(successful_candidates),
                "total_successful_models": len(successful_candidates) + 1,
                "mandatory_model": args.model,
                "mandatory_model_passed": True,
                "attempts": matrix,
            }
            if len(successful_candidates) < args.minimum_model_successes:
                raise LiveCheckFailure(
                    "model_matrix",
                    f"Only {len(successful_candidates)} candidate models passed; "
                    f"required {args.minimum_model_successes} in addition to {args.model}",
                )
        summary["status"] = "passed"
        summary["finished_at"] = int(time.time())
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0
    except LiveCheckFailure as exc:
        summary["stage"] = exc.stage
        summary["error"] = redact_text(exc)
    except Exception as exc:
        summary["stage"] = "unexpected"
        summary["error"] = redact_text(exc)
    summary["finished_at"] = int(time.time())
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
