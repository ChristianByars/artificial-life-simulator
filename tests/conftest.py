from pathlib import Path

import pytest


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the app's user-data directory at a temp dir (M8: ALIFE_DATA_DIR)."""
    monkeypatch.setenv("ALIFE_DATA_DIR", str(tmp_path))
    return tmp_path
