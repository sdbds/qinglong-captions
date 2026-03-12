import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


from utils.lance_utils import (
    build_version_tag,
    get_latest_version_number,
    sanitize_tag_component,
    update_or_create_tag,
)


def test_sanitize_tag_component_normalizes_and_defaults():
    assert sanitize_tag_component(" zh cn / test ") == "zh_cn_test"
    assert sanitize_tag_component("...") == "default"


def test_build_version_tag_uses_fixed_timestamp():
    tag = build_version_tag("tr", "Qwen/Qwen3.5-2B", "zh-cn", timestamp="20260312_100000")
    assert tag == "tr.Qwen_Qwen3.5-2B.zh-cn.20260312_100000"


def test_get_latest_version_number_supports_int_and_versions_list():
    assert get_latest_version_number(SimpleNamespace(version=7)) == 7

    version_obj = SimpleNamespace(version=5)
    dataset = SimpleNamespace(versions=lambda: [{"version": 2}, version_obj, {"version": "bad"}])
    assert get_latest_version_number(dataset) == 5


def test_update_or_create_tag_falls_back_to_update():
    calls = []

    class Tags:
        def create(self, tag, version):
            calls.append(("create", tag, version))
            raise RuntimeError("exists")

        def update(self, tag, version):
            calls.append(("update", tag, version))

    dataset = SimpleNamespace(version=3, tags=Tags())
    version = update_or_create_tag(dataset, "tr.test")

    assert version == 3
    assert calls == [("create", "tr.test", 3), ("update", "tr.test", 3)]
