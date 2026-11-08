import importlib
import sys
import textwrap
from pathlib import Path

import pytest

import alife.lifeforms as lifeforms
from alife.lifeforms import (
    OUR_ASSUMPTION,
    InfoNote,
    Param,
    default_params,
    registry,
)


@pytest.fixture
def extra_packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Make tmp_path a second search dir for alife.lifeforms subpackages."""
    monkeypatch.setattr(lifeforms, "__path__", [*lifeforms.__path__, str(tmp_path)])
    importlib.invalidate_caches()
    yield tmp_path
    for name in [m for m in sys.modules if m.startswith("alife.lifeforms.tmp_")]:
        del sys.modules[name]


def make_package(root: Path, name: str, body: str) -> None:
    pkg = root / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text(textwrap.dedent(body))


def lifeform_body(lf_id: str, name: str) -> str:
    return f"""
        class L:
            id = "{lf_id}"
            name = "{name}"
        LIFEFORM = L
    """


def test_demo_registered():
    reg = registry()
    assert "demo" in reg and reg["demo"].name == "Random walkers (dev)"


def test_cell_not_registered_yet():
    assert "cell" not in registry()


def test_params_valid():
    for cls in registry().values():
        for p in cls.params.values():
            assert p.min <= p.default <= p.max and p.source


def test_ordered_by_name():
    names = [cls.name for cls in registry().values()]
    assert names == sorted(names)


def test_default_params():
    assert default_params(registry()["demo"]) == {"count": 30.0}


def test_broken_package_skipped(extra_packages, capsys):
    make_package(extra_packages, "tmp_broken", "raise RuntimeError('boom')")
    reg = registry()
    assert "demo" in reg
    assert not any("broken" in k for k in reg)
    err = capsys.readouterr().err
    assert "tmp_broken" in err and "boom" in err


def test_package_without_lifeform_skipped(extra_packages):
    make_package(extra_packages, "tmp_plain", "X = 1")
    assert set(registry()) == {"demo"}


def test_extra_package_registered_and_sorted(extra_packages):
    make_package(extra_packages, "tmp_extra", lifeform_body("extra", "AAA first"))
    reg = registry()
    assert reg["extra"].name == "AAA first"
    assert list(reg)[0] == "extra"


def test_duplicate_id_raises(extra_packages):
    make_package(extra_packages, "tmp_dup", lifeform_body("demo", "Impostor"))
    with pytest.raises(ValueError, match="demo"):
        registry()


def test_param_rejects_default_out_of_range():
    with pytest.raises(ValueError):
        Param("x", 5, 0, 1, "our assumption")
    with pytest.raises(ValueError):
        Param("x", -1, 0, 1, "our assumption")


def test_param_rejects_inverted_range_and_nan():
    with pytest.raises(ValueError):
        Param("x", 0.5, 1, 0, OUR_ASSUMPTION)
    with pytest.raises(ValueError):
        Param("x", float("nan"), 0, 1, OUR_ASSUMPTION)


def test_param_requires_source():
    with pytest.raises(ValueError):
        Param("x", 0.5, 0, 1, "")
    with pytest.raises(ValueError):
        Param("x", 0.5, 0, 1, "   ")


def test_param_clamp():
    p = Param("x", 5, 0, 10, OUR_ASSUMPTION, unit="mm")
    assert p.clamp(-3) == 0.0
    assert p.clamp(99) == 10.0
    assert p.clamp(4) == 4.0
    assert p.unit == "mm"


def test_infonote_defaults():
    assert InfoNote("t", "x").assumed is False
