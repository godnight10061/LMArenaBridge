import subprocess
import sys
from pathlib import Path


def test_main_can_run_as_script() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    main_path = src_dir / "main.py"

    code = "\n".join(
        [
            "import runpy, sys, uvicorn",
            "uvicorn.run = lambda *a, **k: None",
            f"repo_root = {str(repo_root)!r}",
            f"src_dir = {str(src_dir)!r}",
            # Mimic `python src/main.py` by putting `.../src` first and removing the repo root.
            "sys.path = [src_dir] + [p for p in sys.path if p and p not in (repo_root, src_dir)]",
            f"runpy.run_path({str(main_path)!r}, run_name='__main__')",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        capture_output=True,
        timeout=60,
    )

    if result.returncode != 0:
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise AssertionError(
            f"script execution failed (rc={result.returncode})\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
