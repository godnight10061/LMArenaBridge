import time
import unittest

from src.challenge_recovery import (
    CHALLENGE_COOLDOWN_SECONDS,
    CHALLENGE_STATE_SCHEMA,
    CHALLENGE_STATE_TTL_SECONDS,
    exhaust,
    fingerprint_request,
    normalize_state,
    preflight,
    record_challenge,
    record_success,
)

MODEL = "gemini-3.5-flash"
PROMPT_A = "first rendered transcript"
PROMPT_B = "unrelated rendered transcript"
FINGERPRINT_A = fingerprint_request(model=MODEL, prompt=PROMPT_A)
FINGERPRINT_B = fingerprint_request(model=MODEL, prompt=PROMPT_B)


class ChallengeRecoveryStateTests(unittest.TestCase):
    def test_request_fingerprint_is_deterministic_and_prompt_sensitive(self) -> None:
        self.assertEqual(
            FINGERPRINT_A,
            fingerprint_request(model=MODEL, prompt=PROMPT_A),
        )
        self.assertNotEqual(FINGERPRINT_A, FINGERPRINT_B)
        self.assertEqual(len(FINGERPRINT_A), 64)
        self.assertNotIn(PROMPT_A, FINGERPRINT_A)

    def test_first_challenge_enters_request_scoped_cooldown(self) -> None:
        now = int(time.time())

        transition = record_challenge(
            None,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now,
        )

        self.assertEqual(transition.action, "respond")
        self.assertEqual(transition.error_code, "challenge_unresolved")
        self.assertEqual(transition.phase, "cooldown")
        self.assertEqual(transition.retry_after, CHALLENGE_COOLDOWN_SECONDS)
        self.assertEqual(transition.state["schema"], CHALLENGE_STATE_SCHEMA)
        self.assertEqual(transition.state["request_fingerprint"], FINGERPRINT_A)
        self.assertEqual(transition.state["retry_not_before"], now + 900)
        self.assertFalse(transition.state["replacement_used"])

    def test_cooldown_then_same_account_retry_then_replacement(self) -> None:
        now = int(time.time())
        first = record_challenge(
            None,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now,
        )

        waiting = preflight(
            first.state,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now + 12,
            replace_requested=False,
        )
        self.assertEqual(waiting.action, "respond")
        self.assertEqual(waiting.phase, "cooldown")
        self.assertEqual(waiting.retry_after, 888)

        retry = preflight(
            first.state,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now + CHALLENGE_COOLDOWN_SECONDS,
            replace_requested=False,
        )
        self.assertEqual(retry.action, "run")
        self.assertEqual(retry.state["phase"], "same_account_retry")

        second = record_challenge(
            retry.state,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now + CHALLENGE_COOLDOWN_SECONDS + 1,
        )
        self.assertEqual(second.action, "respond")
        self.assertEqual(second.phase, "replacement_required")
        self.assertEqual(second.retry_after, 0)

        replacement = preflight(
            second.state,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now + CHALLENGE_COOLDOWN_SECONDS + 2,
            replace_requested=True,
        )
        self.assertEqual(replacement.action, "force_signup")
        self.assertEqual(replacement.state["phase"], "final_attempt")
        self.assertTrue(replacement.state["replacement_used"])

        final = record_challenge(
            replacement.state,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now + CHALLENGE_COOLDOWN_SECONDS + 3,
        )
        self.assertEqual(final.error_code, "challenge_exhausted")
        self.assertEqual(final.phase, "exhausted")

    def test_unrelated_same_model_request_bypasses_cooldown(self) -> None:
        now = int(time.time())
        first = record_challenge(
            None,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now,
        )

        unrelated = preflight(
            first.state,
            model=MODEL,
            request_fingerprint=FINGERPRINT_B,
            now=now + 1,
            replace_requested=False,
        )

        self.assertEqual(unrelated.action, "run")
        self.assertEqual(unrelated.state, first.state)
        self.assertFalse(unrelated.changed)

    def test_unrelated_request_cannot_consume_replacement_budget(self) -> None:
        now = int(time.time())
        first = record_challenge(
            None,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now,
        )
        retry = preflight(
            first.state,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now + CHALLENGE_COOLDOWN_SECONDS,
            replace_requested=False,
        )
        replacement_required = record_challenge(
            retry.state,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now + CHALLENGE_COOLDOWN_SECONDS + 1,
        )

        unrelated = preflight(
            replacement_required.state,
            model=MODEL,
            request_fingerprint=FINGERPRINT_B,
            now=now + CHALLENGE_COOLDOWN_SECONDS + 2,
            replace_requested=True,
        )

        self.assertEqual(unrelated.action, "respond")
        self.assertEqual(unrelated.error_code, "replacement_not_ready")
        self.assertEqual(unrelated.phase, "request_mismatch")
        self.assertEqual(unrelated.state, replacement_required.state)
        self.assertFalse(unrelated.state["replacement_used"])

    def test_unrelated_challenge_replaces_record_without_inheriting_budget(self) -> None:
        now = int(time.time())
        first = record_challenge(
            None,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now,
        )
        retry = preflight(
            first.state,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now + CHALLENGE_COOLDOWN_SECONDS,
            replace_requested=False,
        )
        replacement_required = record_challenge(
            retry.state,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now + CHALLENGE_COOLDOWN_SECONDS + 1,
        )

        unrelated_challenge = record_challenge(
            replacement_required.state,
            model=MODEL,
            request_fingerprint=FINGERPRINT_B,
            now=now + CHALLENGE_COOLDOWN_SECONDS + 2,
        )

        self.assertEqual(unrelated_challenge.phase, "cooldown")
        self.assertEqual(unrelated_challenge.state["request_fingerprint"], FINGERPRINT_B)
        self.assertFalse(unrelated_challenge.state["replacement_used"])

    def test_unrelated_exhaust_does_not_advance_active_record(self) -> None:
        now = int(time.time())
        first = record_challenge(
            None,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now,
        )

        unrelated = exhaust(
            first.state,
            model=MODEL,
            request_fingerprint=FINGERPRINT_B,
            now=now + 1,
        )

        self.assertEqual(unrelated.state, first.state)
        self.assertFalse(unrelated.changed)

    def test_success_clears_only_matching_request_state(self) -> None:
        now = int(time.time())
        first = record_challenge(
            None,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now,
        )

        cleared = record_success(
            first.state,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now + 1,
        )
        self.assertIsNone(cleared.state)
        self.assertTrue(cleared.changed)

        preserved = record_success(
            first.state,
            model=MODEL,
            request_fingerprint=FINGERPRINT_B,
            now=now + 1,
        )
        self.assertEqual(preserved.state, first.state)
        self.assertFalse(preserved.changed)

    def test_expired_state_is_discarded(self) -> None:
        now = int(time.time())
        first = record_challenge(
            None,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now,
        )

        transition = preflight(
            first.state,
            model=MODEL,
            request_fingerprint=FINGERPRINT_A,
            now=now + CHALLENGE_STATE_TTL_SECONDS + 1,
            replace_requested=False,
        )

        self.assertEqual(transition.action, "run")
        self.assertIsNone(transition.state)
        self.assertTrue(transition.changed)

    def test_schema_one_model_wide_state_is_discarded(self) -> None:
        now = int(time.time())
        legacy = {
            "schema": 1,
            "phase": "cooldown",
            "model": MODEL,
            "first_challenge_at": now,
            "retry_not_before": now + 900,
            "replacement_used": False,
        }

        transition = preflight(
            legacy,
            model=MODEL,
            request_fingerprint=FINGERPRINT_B,
            now=now + 1,
            replace_requested=False,
        )

        self.assertEqual(transition.action, "run")
        self.assertIsNone(transition.state)
        self.assertTrue(transition.changed)

    def test_normalization_keeps_metadata_only_and_parses_false_string(self) -> None:
        now = int(time.time())
        raw = {
            "schema": CHALLENGE_STATE_SCHEMA,
            "phase": "replacement_required",
            "model": MODEL,
            "request_fingerprint": FINGERPRINT_A,
            "first_challenge_at": now,
            "retry_not_before": 0,
            "replacement_used": "false",
            "prompt": "must be discarded",
        }

        normalized = normalize_state(raw, now=now)

        self.assertEqual(
            set(normalized or {}),
            {
                "schema",
                "phase",
                "model",
                "request_fingerprint",
                "first_challenge_at",
                "retry_not_before",
                "replacement_used",
            },
        )
        self.assertFalse(normalized["replacement_used"])


if __name__ == "__main__":
    unittest.main()
