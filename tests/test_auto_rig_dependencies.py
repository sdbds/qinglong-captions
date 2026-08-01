from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent


def test_auto_rig_extra_pins_the_geometry_and_psd_runtime() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["optional-dependencies"]["auto-rig"]

    assert dependencies == [
        "numpy==1.26.4",
        "scipy==1.15.3",
        "scikit-image==0.25.2",
        "psd-tools[composite]==1.17.4",
        "pillow==12.3.0",
        "rectpack==0.2.2",
    ]
    assert not any("opencv-contrib" in dependency for dependency in dependencies)
