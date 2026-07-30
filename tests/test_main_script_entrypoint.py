import subprocess
import sys
import unittest
from pathlib import Path


class TestMainScriptEntrypoint(unittest.TestCase):
    def test_main_can_run_as_script(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "src" / "main.py"

        code = "\n".join(
            [
                "import runpy, uvicorn",
                "uvicorn.run = lambda *a, **k: None",
                f"runpy.run_path({str(main_path)!r}, run_name='__main__')",
            ]
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(repo_root),
            capture_output=True,
        )

        if result.returncode != 0:
            stdout = result.stdout.decode("utf-8", errors="replace")
            stderr = result.stderr.decode("utf-8", errors="replace")
            self.fail(f"script execution failed (rc={result.returncode})\nstdout:\n{stdout}\nstderr:\n{stderr}")

