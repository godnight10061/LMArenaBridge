"""Pure persisted state transitions for bounded Arena challenge recovery."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

CHALLENGE_STATE_SCHEMA = 1
CHALLENGE_COOLDOWN_SECONDS = 900
CHALLENGE_STATE_TTL_SECONDS = 24 * 60 * 60
CHALLENGE_ERROR_HEADER = "X-LMBridge-Error-Code"
CHALLENGE_PHASE_HEADER = "X-LMBridge-Challenge-Phase"
REPLACE_ACCOUNT_HEADER = "X-LMBridge-Replace-Account"

ChallengePhase = Literal[
    "cooldown",
    "same_account_retry",
    "replacement_required",
    "final_attempt",
    "exhausted",
]
ChallengeAction = Literal["run", "respond", "force_signup"]

_VALID_PHASES = {
    "cooldown",
    "same_account_retry",
    "replacement_required",
    "final_attempt",
    "exhausted",
}


@dataclass(frozen=True)
class ChallengeTransition:
    action: ChallengeAction
    state: dict[str, Any] | None
    error_code: str = ""
    phase: str = ""
    retry_after: int = 0
    changed: bool = False


def _as_timestamp(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        timestamp = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return timestamp if timestamp > 0 else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _normalize(value: Any, *, now: float) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(value, dict):
        return None, value is not None
    if value.get("schema") != CHALLENGE_STATE_SCHEMA:
        return None, bool(value)

    phase = str(value.get("phase") or "").strip()
    model = str(value.get("model") or "").strip()
    first_challenge_at = _as_timestamp(value.get("first_challenge_at"))
    retry_not_before = _as_timestamp(value.get("retry_not_before")) or 0
    if phase not in _VALID_PHASES or not model or first_challenge_at is None:
        return None, True
    if float(now) - first_challenge_at >= CHALLENGE_STATE_TTL_SECONDS:
        return None, True

    normalized = {
        "schema": CHALLENGE_STATE_SCHEMA,
        "phase": phase,
        "model": model,
        "first_challenge_at": first_challenge_at,
        "retry_not_before": retry_not_before,
        "replacement_used": _as_bool(value.get("replacement_used", False)),
    }
    return normalized, normalized != value


def normalize_state(value: Any, *, now: float) -> dict[str, Any] | None:
    """Return a metadata-only valid state, dropping invalid or expired input."""
    state, _changed = _normalize(value, now=now)
    return state


def _remaining_cooldown(state: dict[str, Any], *, now: float) -> int:
    return max(0, int(math.ceil(float(state["retry_not_before"]) - float(now))))


def _response(
    state: dict[str, Any] | None,
    *,
    error_code: str,
    phase: str,
    retry_after: int = 0,
    changed: bool = False,
) -> ChallengeTransition:
    return ChallengeTransition(
        action="respond",
        state=state,
        error_code=error_code,
        phase=phase,
        retry_after=max(0, int(retry_after)),
        changed=changed,
    )


def preflight(
    value: Any,
    *,
    model: str,
    now: float,
    replace_requested: bool,
) -> ChallengeTransition:
    """Decide whether a request runs, waits, or consumes the replacement budget."""
    state, changed = _normalize(value, now=now)
    requested_model = str(model or "").strip()
    if state is None:
        if replace_requested:
            return _response(
                None,
                error_code="replacement_not_ready",
                phase="none",
                changed=changed,
            )
        return ChallengeTransition(action="run", state=None, changed=changed)
    if state["model"] != requested_model:
        if replace_requested:
            return _response(
                state,
                error_code="replacement_not_ready",
                phase=str(state["phase"]),
                changed=changed,
            )
        return ChallengeTransition(action="run", state=state, changed=changed)

    phase = str(state["phase"])
    if replace_requested:
        if phase == "replacement_required" and not state["replacement_used"]:
            updated = dict(state)
            updated.update(
                {
                    "phase": "final_attempt",
                    "retry_not_before": 0,
                    "replacement_used": True,
                }
            )
            return ChallengeTransition(
                action="force_signup",
                state=updated,
                phase="final_attempt",
                changed=True,
            )
        return _response(
            state,
            error_code="replacement_not_ready",
            phase=phase,
            retry_after=_remaining_cooldown(state, now=now) if phase == "cooldown" else 0,
            changed=changed,
        )

    if phase == "cooldown":
        remaining = _remaining_cooldown(state, now=now)
        if remaining > 0:
            return _response(
                state,
                error_code="challenge_unresolved",
                phase="cooldown",
                retry_after=remaining,
                changed=changed,
            )
        updated = dict(state)
        updated.update({"phase": "same_account_retry", "retry_not_before": 0})
        return ChallengeTransition(
            action="run",
            state=updated,
            phase="same_account_retry",
            changed=True,
        )
    if phase == "same_account_retry":
        # A persisted in-flight retry means the process ended without a completed
        # transition. Preserve the one-retry bound and move to replacement gating.
        updated = dict(state)
        updated["phase"] = "replacement_required"
        return _response(
            updated,
            error_code="challenge_unresolved",
            phase="replacement_required",
            changed=True,
        )
    if phase == "replacement_required":
        return _response(
            state,
            error_code="challenge_unresolved",
            phase="replacement_required",
            changed=changed,
        )
    if phase == "final_attempt":
        updated = dict(state)
        updated["phase"] = "exhausted"
        return _response(
            updated,
            error_code="challenge_exhausted",
            phase="exhausted",
            changed=True,
        )
    return _response(
        state,
        error_code="challenge_exhausted",
        phase="exhausted",
        changed=changed,
    )


def record_challenge(value: Any, *, model: str, now: float) -> ChallengeTransition:
    """Persist the next bounded phase after a real unresolved browser challenge."""
    state, changed = _normalize(value, now=now)
    requested_model = str(model or "").strip()
    if state is None or state["model"] != requested_model:
        timestamp = int(float(now))
        updated = {
            "schema": CHALLENGE_STATE_SCHEMA,
            "phase": "cooldown",
            "model": requested_model,
            "first_challenge_at": timestamp,
            "retry_not_before": timestamp + CHALLENGE_COOLDOWN_SECONDS,
            "replacement_used": False,
        }
        return _response(
            updated,
            error_code="challenge_unresolved",
            phase="cooldown",
            retry_after=CHALLENGE_COOLDOWN_SECONDS,
            changed=True,
        )

    phase = str(state["phase"])
    if phase == "same_account_retry":
        updated = dict(state)
        updated["phase"] = "replacement_required"
        return _response(
            updated,
            error_code="challenge_unresolved",
            phase="replacement_required",
            changed=True,
        )
    if phase == "final_attempt":
        updated = dict(state)
        updated["phase"] = "exhausted"
        return _response(
            updated,
            error_code="challenge_exhausted",
            phase="exhausted",
            changed=True,
        )
    if phase == "cooldown":
        return _response(
            state,
            error_code="challenge_unresolved",
            phase="cooldown",
            retry_after=max(1, _remaining_cooldown(state, now=now)),
            changed=changed,
        )
    if phase == "replacement_required":
        return _response(
            state,
            error_code="challenge_unresolved",
            phase="replacement_required",
            changed=changed,
        )
    return _response(
        state,
        error_code="challenge_exhausted",
        phase="exhausted",
        changed=changed,
    )


def exhaust(value: Any, *, model: str, now: float) -> ChallengeTransition:
    """Consume the recovery budget after a terminal replacement/final-attempt failure."""
    state, changed = _normalize(value, now=now)
    if state is None or state["model"] != str(model or "").strip():
        return ChallengeTransition(action="respond", state=state, changed=changed)
    if state["phase"] == "exhausted":
        return _response(
            state,
            error_code="challenge_exhausted",
            phase="exhausted",
            changed=changed,
        )
    updated = dict(state)
    updated["phase"] = "exhausted"
    updated["replacement_used"] = True
    updated["retry_not_before"] = 0
    return _response(
        updated,
        error_code="challenge_exhausted",
        phase="exhausted",
        changed=True,
    )


def record_success(value: Any, *, model: str, now: float) -> ChallengeTransition:
    """Clear matching challenge metadata after a successful model response."""
    state, changed = _normalize(value, now=now)
    if state is not None and state["model"] == str(model or "").strip():
        return ChallengeTransition(action="run", state=None, changed=True)
    return ChallengeTransition(action="run", state=state, changed=changed)
