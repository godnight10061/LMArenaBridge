import time
import unittest

from src.challenge_recovery import (
    CHALLENGE_COOLDOWN_SECONDS,
    CHALLENGE_STATE_TTL_SECONDS,
    normalize_state,
    preflight,
    record_challenge,
    record_success,
)

MODEL = "gemini-3.5-flash"


class ChallengeRecoveryStateTests(unittest.TestCase):
    def test_first_challenge_enters_cooldown(self) -> None:
        now = int(time.time())

        transition = record_challenge(None, model=MODEL, now=now)

        self.assertEqual(transition.action, "respond")
        self.assertEqual(transition.error_code, "challenge_unresolved")
        self.assertEqual(transition.phase, "cooldown")
        self.assertEqual(transition.retry_after, CHALLENGE_COOLDOWN_SECONDS)
        self.assertEqual(transition.state["retry_not_before"], now + 900)
        self.assertFalse(transition.state["replacement_used"])

    def test_cooldown_then_same_account_retry_then_replacement(self) -> None:
        now = int(time.time())
        first = record_challenge(None, model=MODEL, now=now)

        waiting = preflight(
            first.state,
            model=MODEL,
            now=now + 12,
            replace_requested=False,
        )
        self.assertEqual(waiting.action, "respond")
        self.assertEqual(waiting.phase, "cooldown")
        self.assertEqual(waiting.retry_after, 888)

        retry = preflight(
            first.state,
            model=MODEL,
            now=now + CHALLENGE_COOLDOWN_SECONDS,
            replace_requested=False,
        )
        self.assertEqual(retry.action, "run")
        self.assertEqual(retry.state["phase"], "same_account_retry")

        second = record_challenge(
            retry.state,
            model=MODEL,
            now=now + CHALLENGE_COOLDOWN_SECONDS + 1,
        )
        self.assertEqual(second.action, "respond")
        self.assertEqual(second.phase, "replacement_required")
        self.assertEqual(second.retry_after, 0)

        replacement = preflight(
            second.state,
            model=MODEL,
            now=now + CHALLENGE_COOLDOWN_SECONDS + 2,
            replace_requested=True,
        )
        self.assertEqual(replacement.action, "force_signup")
        self.assertEqual(replacement.state["phase"], "final_attempt")
        self.assertTrue(replacement.state["replacement_used"])

        final = record_challenge(
            replacement.state,
            model=MODEL,
            now=now + CHALLENGE_COOLDOWN_SECONDS + 3,
        )
        self.assertEqual(final.error_code, "challenge_exhausted")
        self.assertEqual(final.phase, "exhausted")

    def test_success_clears_only_matching_model_state(self) -> None:
        now = int(time.time())
        first = record_challenge(None, model=MODEL, now=now)

        cleared = record_success(first.state, model=MODEL, now=now + 1)
        self.assertIsNone(cleared.state)
        self.assertTrue(cleared.changed)

        preserved = record_success(first.state, model="another-model", now=now + 1)
        self.assertEqual(preserved.state, first.state)
        self.assertFalse(preserved.changed)

    def test_expired_state_is_discarded(self) -> None:
        now = int(time.time())
        first = record_challenge(None, model=MODEL, now=now)

        transition = preflight(
            first.state,
            model=MODEL,
            now=now + CHALLENGE_STATE_TTL_SECONDS + 1,
            replace_requested=False,
        )

        self.assertEqual(transition.action, "run")
        self.assertIsNone(transition.state)
        self.assertTrue(transition.changed)

    def test_normalization_keeps_metadata_only_and_parses_false_string(self) -> None:
        now = int(time.time())
        raw = {
            "schema": 1,
            "phase": "replacement_required",
            "model": MODEL,
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
                "first_challenge_at",
                "retry_not_before",
                "replacement_used",
            },
        )
        self.assertFalse(normalized["replacement_used"])


if __name__ == "__main__":
    unittest.main()
