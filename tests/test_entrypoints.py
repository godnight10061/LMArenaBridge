import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_health(process: subprocess.Popen, port: int) -> dict:
    deadline = time.monotonic() + 30
    url = f"http://127.0.0.1:{port}/api/v1/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"bridge exited before health check (exit={process.returncode})\n"
                f"stdout={stdout[-2000:]}\nstderr={stderr[-2000:]}"
            )
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(0.1)
    raise AssertionError(f"bridge did not become healthy at {url}")


class TestBridgeEntrypoints(unittest.TestCase):
    def test_direct_file_and_module_launch_bootstrap_the_same_server(self) -> None:
        commands = (
            [sys.executable, "src/main.py"],
            [sys.executable, "-m", "src.main"],
        )
        for command in commands:
            with (
                self.subTest(command=command),
                tempfile.TemporaryDirectory(prefix="lmarena-entrypoint-") as runtime_dir,
            ):
                port = _reserve_loopback_port()
                environment = dict(os.environ)
                environment.update(
                    {
                        "LM_CONFIG_FILE": str(Path(runtime_dir) / "config.json"),
                        "LM_MODELS_FILE": str(Path(runtime_dir) / "models.json"),
                        "PORT": str(port),
                        "PYTEST_CURRENT_TEST": "entrypoint_subprocess",
                        "PYTHONUNBUFFERED": "1",
                    }
                )
                process = subprocess.Popen(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    health = _wait_for_health(process, port)
                    self.assertIn(health.get("status"), {"healthy", "degraded"})
                    self.assertIsNone(process.poll())
                finally:
                    process.terminate()
                    try:
                        process.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
