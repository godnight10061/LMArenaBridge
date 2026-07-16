import os
import sys
import shutil
from pathlib import Path

import pytest


def _norm_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path or os.getcwd()))


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_NORM = _norm_path(str(REPO_ROOT))

if not any(_norm_path(p) == REPO_ROOT_NORM for p in sys.path):
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session", autouse=True)
def isolate_runtime_files(tmp_path_factory):
    """Keep tests from overwriting the bridge's real config and model cache."""
    project_root = Path.cwd()
    runtime_dir = tmp_path_factory.mktemp("lmarena-bridge-runtime")

    for name in ("config.json", "models.json"):
        source = project_root / name
        if source.is_file():
            shutil.copy2(source, runtime_dir / name)

    os.chdir(runtime_dir)
    try:
        yield
    finally:
        os.chdir(project_root)
