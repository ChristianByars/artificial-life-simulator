"""F3: atomic save store with validation (SPEC §4)."""

import json
from pathlib import Path

import pytest

from alife import save
from alife.save import SAVE_KEYS, SCHEMA_VERSION, SaveError, SaveStore, saves_dir


def make_save(lifeform_id: str = "demo", **overrides: object) -> dict:
    data: dict = {
        "schema_version": SCHEMA_VERSION,
        "app_version": "0.3.0",
        "lifeform_id": lifeform_id,
        "saved_at": "2026-11-07T16:00:00-08:00",
        "seed": 12345,
        "rng_state": [3, [1, 2, 3], None],
        "params": {"a": 1.5},
        "tick": 5400,
        "speed": 1.0,
        "camera": {"position": [0, 8, -12], "rotation": [35, 0, 0]},
        "state": {"e": 0.5},
    }
    data.update(overrides)
    return data


def test_save_keys_match_spec() -> None:
    assert set(SAVE_KEYS) == set(make_save())
    assert len(SAVE_KEYS) == 11


def test_saves_dir_uses_env(data_dir: Path) -> None:
    assert saves_dir() == data_dir / "saves"


def test_saves_dir_default_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALIFE_DATA_DIR", raising=False)
    assert saves_dir().name == "saves"
    assert saves_dir().parent.name != ""


def test_write_read_roundtrip(data_dir: Path) -> None:
    store = SaveStore()
    data = make_save()
    path = store.write("demo", data)
    assert path == data_dir / "saves" / "demo.json"
    assert store.exists("demo")
    result = store.read("demo")
    assert result.status == "ok"
    assert result.data == data
    assert result.message == save.MSG_OK


def test_explicit_root(tmp_path: Path) -> None:
    store = SaveStore(tmp_path / "custom")
    store.write("demo", make_save())
    assert (tmp_path / "custom" / "demo.json").exists()


def test_write_is_atomic(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SaveStore()
    old = make_save(tick=1)
    store.write("demo", old)

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr(save.json, "dump", boom)
        with pytest.raises(SaveError):
            store.write("demo", make_save(tick=2))

    assert store.read("demo").data == old
    assert list((data_dir / "saves").glob("*.tmp")) == []
    assert [p.name for p in (data_dir / "saves").iterdir()] == ["demo.json"]


def test_nan_rejected(data_dir: Path) -> None:
    store = SaveStore()
    old = make_save()
    store.write("demo", old)
    with pytest.raises(SaveError):
        store.write("demo", make_save(state={"e": float("nan")}))
    assert store.read("demo").data == old
    assert list((data_dir / "saves").glob("*.tmp")) == []


def test_unserializable_rejected(data_dir: Path) -> None:
    store = SaveStore()
    with pytest.raises(SaveError):
        store.write("demo", make_save(state={"x": object()}))
    assert not store.exists("demo")


def test_read_missing(data_dir: Path) -> None:
    result = SaveStore().read("demo")
    assert result.status == "missing"
    assert result.data is None
    assert result.message == save.MSG_MISSING


def test_read_truncated(data_dir: Path) -> None:
    store = SaveStore()
    text = json.dumps(make_save())
    store.path("demo").parent.mkdir(parents=True)
    store.path("demo").write_text(text[: len(text) // 2], encoding="utf-8")
    result = store.read("demo")
    assert result.status == "corrupt"
    assert result.data is None
    assert result.message == save.MSG_CORRUPT


def test_read_empty_file(data_dir: Path) -> None:
    store = SaveStore()
    store.path("demo").parent.mkdir(parents=True)
    store.path("demo").write_text("", encoding="utf-8")
    assert store.read("demo").status == "corrupt"


def test_read_binary_garbage(data_dir: Path) -> None:
    store = SaveStore()
    store.path("demo").parent.mkdir(parents=True)
    store.path("demo").write_bytes(b"\xff\xfe\x00\x80garbage")
    assert store.read("demo").status == "corrupt"


def test_read_not_object(data_dir: Path) -> None:
    store = SaveStore()
    store.path("demo").parent.mkdir(parents=True)
    store.path("demo").write_text("[]", encoding="utf-8")
    assert store.read("demo").status == "corrupt"


def test_read_missing_key(data_dir: Path) -> None:
    store = SaveStore()
    data = make_save()
    del data["state"]
    store.path("demo").parent.mkdir(parents=True)
    store.path("demo").write_text(json.dumps(data), encoding="utf-8")
    assert store.read("demo").status == "corrupt"


def test_read_unknown_schema(data_dir: Path) -> None:
    store = SaveStore()
    store.write("demo", make_save(schema_version=2))
    result = store.read("demo")
    assert result.status == "unknown_schema"
    assert result.data is None
    assert result.message == save.MSG_UNKNOWN_SCHEMA


def test_read_wrong_lifeform(data_dir: Path) -> None:
    store = SaveStore()
    store.write("demo", make_save(lifeform_id="cell"))
    result = store.read("demo")
    assert result.status == "wrong_lifeform"
    assert result.message == save.MSG_WRONG_LIFEFORM


def test_corrupt_checked_before_schema(data_dir: Path) -> None:
    store = SaveStore()
    data = make_save(schema_version=2)
    del data["tick"]
    store.path("demo").parent.mkdir(parents=True)
    store.path("demo").write_text(json.dumps(data), encoding="utf-8")
    assert store.read("demo").status == "corrupt"


def test_schema_checked_before_lifeform(data_dir: Path) -> None:
    store = SaveStore()
    store.write("demo", make_save(schema_version=2, lifeform_id="cell"))
    assert store.read("demo").status == "unknown_schema"


def test_creates_saves_dir(data_dir: Path) -> None:
    assert not (data_dir / "saves").exists()
    SaveStore().write("demo", make_save())
    assert (data_dir / "saves").is_dir()


def test_overwrite_replaces(data_dir: Path) -> None:
    store = SaveStore()
    store.write("demo", make_save(tick=1))
    store.write("demo", make_save(tick=2))
    assert store.read("demo").data["tick"] == 2  # type: ignore[index]
