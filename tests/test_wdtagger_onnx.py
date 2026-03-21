import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def test_wdtagger_load_model_and_tags_uses_single_model_bundle(monkeypatch, tmp_path):
    sys.modules.pop("utils.wdtagger", None)

    fake_torch = types.ModuleType("torch")
    fake_cv2 = types.ModuleType("cv2")
    fake_lance = types.ModuleType("lance")
    fake_ort = types.ModuleType("onnxruntime")
    fake_hf = types.ModuleType("huggingface_hub")
    fake_lance_import = types.ModuleType("module.lanceImport")

    fake_hf.hf_hub_download = lambda **kwargs: str(tmp_path / kwargs["filename"])
    fake_lance_import.transform2lance = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "lance", fake_lance)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "module.lanceImport", fake_lance_import)

    wdtagger = importlib.import_module("utils.wdtagger")

    captured = {}
    repo_dir = tmp_path / "SmilingWolf_wd-v1-4-moat-tagger-v2"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "selected_tags.csv").write_text(
        "tag_id,name,category,count\n0,safe,9,1\n1,1girl,0,1\n2,character_a,4,1\n",
        encoding="utf-8",
    )

    fake_session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="input")])

    def fake_loader(*, spec, runtime_config):
        captured["spec"] = spec
        captured["runtime"] = runtime_config
        return SimpleNamespace(
            session=fake_session,
            providers=("CPUExecutionProvider",),
            input_metas=(SimpleNamespace(name="input"),),
        )

    monkeypatch.setattr(wdtagger, "load_single_model_bundle", fake_loader, raising=False)

    args = SimpleNamespace(
        repo_id="SmilingWolf/wd-v1-4-moat-tagger-v2",
        model_dir=str(tmp_path),
        force_download=False,
        remove_parents_tag=False,
    )

    session, input_name, label_data, parent_to_child_map = wdtagger.load_model_and_tags(args)

    assert session is fake_session
    assert input_name == "input"
    assert label_data.names[0] == "safe"
    assert parent_to_child_map == {}
    assert captured["spec"].bundle_key == "wdtagger:SmilingWolf/wd-v1-4-moat-tagger-v2"
