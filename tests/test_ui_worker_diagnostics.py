import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestUiWorkerDiagnostics(unittest.TestCase):
    def test_initial_recovery_exception_still_writes_structured_result(self):
        project_root = Path(__file__).resolve().parent.parent
        worker = project_root / "src" / "ui_bridge_worker.py"
        with tempfile.TemporaryDirectory(prefix="lmarena-worker-test-") as temp_dir:
            temp_path = Path(temp_dir)
            invalid_profile = temp_path / "profile-is-a-file"
            invalid_profile.write_text("not a directory", encoding="utf-8")
            result_path = temp_path / "result.json"
            env = dict(os.environ)
            env.update(
                {
                    "LM_CHROME_PROFILE_DIR": str(invalid_profile),
                    "LM_CONFIG_FILE": str(temp_path / "config.json"),
                    "LM_UI_RESULT_PATH": str(result_path),
                    "PYTHONUNBUFFERED": "1",
                }
            )
            completed = subprocess.run(
                [sys.executable, str(worker)],
                input=json.dumps(
                    {
                        "model": "gemini-3.5-flash",
                        "prompt": "diagnostic fixture",
                        "timeout_seconds": 20,
                    }
                ),
                text=True,
                cwd=str(project_root),
                env=env,
                capture_output=True,
                timeout=30,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertIsNone(payload["response"])
            self.assertEqual(payload["stage"], "initial_recovery")
            self.assertEqual(payload["error_code"], "worker_exception")
            self.assertTrue(payload["error_type"])


if __name__ == "__main__":
    unittest.main()
