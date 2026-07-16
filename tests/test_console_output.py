import io
import os
import unittest

from src.console_output import (
    configure_standard_streams,
    render_for_encoding,
    should_force_utf8,
)
from src.main import safe_print


class _TerminalTextIOWrapper(io.TextIOWrapper):
    def isatty(self) -> bool:
        return True


class ConsoleOutputTests(unittest.TestCase):
    def test_only_redirected_streams_force_utf8(self) -> None:
        self.assertTrue(should_force_utf8(is_terminal=False))
        self.assertFalse(should_force_utf8(is_terminal=True))

    def test_redirected_real_text_stream_is_reconfigured_to_utf8(self) -> None:
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")

        configure_standard_streams(stream)
        stream.write("bridge \U0001f680")
        stream.flush()

        self.assertEqual(stream.encoding.lower().replace("-", ""), "utf8")
        self.assertEqual(raw.getvalue().decode("utf-8"), "bridge \U0001f680")

    def test_native_console_fallback_is_readable_without_mojibake(self) -> None:
        rendered = render_for_encoding("\U0001f680 LMArena Bridge", "cp936")

        self.assertEqual(rendered, "[START] LMArena Bridge")
        rendered.encode("cp936")

    def test_real_terminal_stream_keeps_native_encoding_and_safe_prints(self) -> None:
        raw = io.BytesIO()
        stream = _TerminalTextIOWrapper(raw, encoding="cp936", errors="strict")

        configure_standard_streams(stream)
        safe_print("\U0001f680 LMArena Bridge", file=stream, flush=True)

        self.assertEqual(stream.encoding.lower(), "cp936")
        self.assertEqual(raw.getvalue().decode("cp936"), f"[START] LMArena Bridge{os.linesep}")


if __name__ == "__main__":
    unittest.main()
