import hashlib
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageStat


def _tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "deepseek_ocr2_compare.py"
    spec = importlib.util.spec_from_file_location("deepseek_compare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fixture_has_distinct_chinese_pages_and_lossless_deterministic_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    tool = _tool()
    try:
        font = tool.cjk_font()
    except ValueError as exc:
        pytest.skip(str(exc))
    first, second = tmp_path / "one", tmp_path / "two"
    truth = tool.prepare(first, font)
    tool.prepare(second, font)
    pages = truth["pages"]
    assert len(pages) == 2 and pages[0]["marker"] != pages[1]["marker"]
    assert sorted(p.name for p in first.iterdir()) == ["page-1.png", "page-2.png", "pages.pdf", "truth.json"]
    for f in first.iterdir():
        assert (second / f.name).read_bytes() == f.read_bytes()
    with fitz.open(first / "pages.pdf") as doc:
        assert len(doc) == 2
        for i, page in enumerate(pages):
            with Image.open(first / page["file"]) as png:
                pix = doc[i].get_pixmap(dpi=144, alpha=False)
                restored = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                assert ImageChops.difference(png, restored).getbbox() is None
                assert page["marker"] in "".join(page["lines"])
                assert len("".join(c for c in "".join(page["lines"]) if "\u4e00" <= c <= "\u9fff")) >= 25
                assert max(ImageStat.Stat(png.crop(page["figure_box_px"])).stddev) > 30
                inks = [box for box, ink in tool._ink_lines(png, page["figure_box_px"]) if ink.any()]
                assert len(inks) == len(page["lines"]) == 4
                assert inks[2][3] < page["figure_box_px"][1] < page["figure_box_px"][3] < inks[3][1]
    with pytest.raises((ValueError, FileExistsError)):
        tool.prepare(first, font)


def _write_json(file, payload):
    file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _prediction(folder, regions):
    raw = "".join(f"<|ref|>{r['label']}<|/ref|><|det|>[{json.dumps(r['bbox'])}]<|/det|>{r['text']}\n" for r in regions)
    (folder / "raw.txt").write_text(raw + "<｜end▁of▁sentence｜>", encoding="utf-8")


def _source_regions(lines):
    rects = [[70, 13, 380, 50], [70, 75, 520, 112], [70, 137, 490, 175], [100, 188, 899, 749], [70, 837, 410, 887]]
    labels = ["title", "text", "text", "image", "image_caption"]
    return [
        dict(
            label=label,
            text=text,
            bbox=b,
            bbox_pixels=[v * a // 999 for v, a in zip(b, (200, 160, 200, 160))],
            **({"crop": "images/0.jpg"} if label == "image" else {}),
        )
        for label, text, b in zip(labels, [*lines[:3], "", lines[3]], rects)
    ]


@pytest.fixture
def captures(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    box = (20, 30, 180, 120)
    texts = [
        ["甲岛观测", "今天潮水退去", "白塔倒映海面", "图一海岛之塔"],
        ["乙山笔记", "这里积雪未化", "湖水清澈见底", "图二山间清湖"],
    ]
    pages, pictures = [], []
    for i, text in enumerate(texts, 1):
        page = Image.new("RGB", (200, 160), "white")
        for ink in ((14, 2, 76, 8), (14, 12, 104, 18), (14, 22, 98, 28), (14, 134, 82, 142)):
            page.paste((35, 35, 35), ink)
        crop = Image.new("RGB", (160, 90))
        crop.putdata([((40 if i == 1 else 200), (x * 2 + y) % 256, (x + 3 * y) % 256) for y in range(90) for x in range(160)])
        page.paste(crop, (20, 30))
        pictures.append(page)
        filename = fixture / f"page-{i}.png"
        page.save(filename)
        pages.append(
            dict(
                lines=text,
                marker=text[0][:2],
                file=filename.name,
                sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
                figure_box_px=box,
            )
        )
    pdf = fixture / "pages.pdf"
    Image.init()
    pictures[0].save(pdf, save_all=True, append_images=pictures[1:])
    _write_json(
        fixture / "truth.json",
        {
            "schema": 1,
            "pages": pages,
            "size": [200, 160],
            "pdf": pdf.name,
            "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        },
    )
    runs = {}
    for name in ("legacy", "native"):
        runs[name] = []
        for n in (1, 2):
            run = tmp_path / f"{name}-{n}"
            runs[name].append(run)
            cases = {}
            for case in ("png", "pdf"):
                cases[case] = []
                for i, text in enumerate(texts):
                    target = run / case / str(i + 1)
                    cases[case].append(f"{case}/{i + 1}")
                    (target / "images").mkdir(parents=True)
                    regions = _source_regions(text)
                    _prediction(target, regions)
                    (target / "result.md").write_text("\n".join([*text[:3], "![](images/0.jpg)", text[3]]), encoding="utf-8")
                    if name == "native":
                        _write_json(
                            target / "grounding.json",
                            dict(
                                schema_version=1,
                                coordinate_system="normalized_0_999",
                                page_number=1 if case == "png" else i + 1,
                                width=200,
                                height=160,
                                regions=regions,
                            ),
                        )
                    pictures[i].crop(regions[3]["bbox_pixels"]).save(target / "images/0.jpg", quality=88)
            _write_json(run / "run.json", {"backend": name, "cases": cases})
    return fixture, runs


def test_compare_measures_independent_runs_and_both_formats(captures):
    fixture, runs = captures
    tool = _tool()
    assert hasattr(tool, "compare"), "Missing offline capture evaluator"
    report = tool.compare(fixture, **runs)
    assert report["passed"], report["errors"]
    assert len(report["runs"]) == 4
    measurements = report["runs"][0]["pages"]
    assert len(measurements) == 4
    assert all(p["text_accuracy"] == 1 and p["box_iou"] > 0.99 and p["crop_pixel_similarity"] > 0.98 for p in measurements)
    assert {"repeat", "backends", "formats"} == {p["kind"] for p in report["comparisons"]}
    assert report["summary"]["measured_pages"] == 16 and report["summary"]["max_character_error_rate"] == 0
    assert set(measurements[0]["text_regions"]) == {"title", "body", "caption"}
    assert measurements[0]["min_region_ink_coverage"] > 0.97
    assert report["runs"][2]["pages"][0]["native_grounding_verified"] is True


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("no_native", "two distinct native runs"),
        ("reuse", "distinct"),
        ("no_page", "two pages"),
        ("no_raw", "raw.txt"),
        ("no_detection", "figure box"),
        ("wrong_location", "box_iou"),
        ("unlinked", "linked crop"),
        ("no_crop", "0.jpg"),
        ("bad_pixels", "crop_pixel_similarity"),
        ("no_chinese", "text_accuracy"),
        ("swapped", "page_order"),
        ("wrong_backend", "backend"),
        ("escape", "relative"),
        ("rerun_drift", "repeat"),
        ("tampered", "hash"),
        ("thumbnail", "crop_resolution_ratio"),
        ("bad_manifest", "cases"),
        ("other_fixture", "fixture hash"),
    ],
)
def test_compare_rejects_incomplete_misaligned_or_drifting_results(captures, change, reason):
    fixture, runs = captures
    second = runs["native"][1]
    page = second / "png/1"
    metadata = second / "run.json"
    capture = json.loads(metadata.read_text(encoding="utf-8"))
    if change == "no_native":
        runs["native"] = []
    elif change == "reuse":
        runs["native"][1] = runs["native"][0]
    elif change == "no_page":
        capture["cases"]["pdf"].pop()
    elif change == "no_raw":
        (page / "raw.txt").unlink()
    elif change in ("no_detection", "wrong_location"):
        (page / "raw.txt").write_text(
            "text" if change == "no_detection" else "<|ref|>figure<|/ref|><|det|>[[10,10,100,100]]<|/det|>", encoding="utf-8"
        )
    elif change == "unlinked":
        (page / "result.md").write_text("甲岛", encoding="utf-8")
    elif change == "no_crop":
        (page / "images/0.jpg").unlink()
    elif change == "bad_pixels":
        Image.new("RGB", (160, 90), "white").save(page / "images/0.jpg")
    elif change == "thumbnail":
        target = page / "images/0.jpg"
        with Image.open(target) as original:
            thumb = original.resize((32, 18))
        thumb.save(target)
    elif change == "no_chinese":
        (page / "result.md").write_text("甲岛 LOREM IPSUM ![](images/0.jpg)", encoding="utf-8")
    elif change == "swapped":
        capture["cases"]["pdf"].reverse()
    elif change == "wrong_backend":
        capture["backend"] = "legacy"
    elif change == "escape":
        (page / "result.md").write_text("![img](../../../../secret.png)", encoding="utf-8")
    elif change == "rerun_drift":
        file = page / "result.md"
        file.write_text(file.read_text(encoding="utf-8").replace("白塔", "石塔"), encoding="utf-8")
    elif change == "tampered":
        (fixture / "page-1.png").write_bytes((fixture / "page-2.png").read_bytes())
    elif change == "bad_manifest":
        capture["cases"] = []
    elif change == "other_fixture":
        capture["fixture_sha256"] = "0" * 64
    _write_json(metadata, capture)
    report = _tool().compare(fixture, **runs)
    assert not report["passed"]
    assert any(reason in error for error in report["errors"]), report["errors"]


def test_cli_writes_failed_report_when_no_model_ran(captures, tmp_path):
    fixture, _ = captures
    out = tmp_path / "comparison.json"
    assert _tool().main(["compare", "--fixture", str(fixture), "--out", str(out)]) != 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert not report["passed"] and report["errors"] and not report["runs"]


@pytest.mark.parametrize(("which", "relocated"), [(0, [70, 75, 520, 112]), (1, [70, 837, 410, 887]), (4, [70, 13, 380, 50])])
def test_correct_ocr_and_figure_cannot_hide_misplaced_grounded_text(captures, which, relocated):
    fixture, runs = captures
    folder = runs["legacy"][1] / "png/1"
    lines = json.loads((fixture / "truth.json").read_text(encoding="utf-8"))["pages"][0]["lines"]
    regions = _source_regions(lines)
    regions[which]["bbox"] = relocated
    _prediction(folder, regions)
    report = _tool().compare(fixture, **runs)
    assert not report["passed"]
    assert any("legacy-2/png/1" in s and "min_region_ink_coverage" in s for s in report["errors"])
    item = report["runs"][1]["pages"][0]
    assert item["text_accuracy"] == 1 and item["box_iou"] > 0.99


def test_two_body_lines_can_share_one_grounded_paragraph(captures):
    fixture, runs = captures
    folder = runs["legacy"][1] / "png/1"
    lines = json.loads((fixture / "truth.json").read_text(encoding="utf-8"))["pages"][0]["lines"]
    regions = _source_regions(lines)
    regions[1]["bbox"] = [70, 75, 520, 175]
    regions[1]["text"] += "\n" + regions.pop(2)["text"]
    _prediction(folder, regions)
    report = _tool().compare(fixture, **runs)
    assert report["passed"], report["errors"]
    assert report["runs"][1]["pages"][0]["text_regions"]["body"]["ink_coverage"] > 0.97


def test_rerun_compares_text_boxes_even_when_both_are_near_ink(captures):
    fixture, runs = captures
    folder = runs["legacy"][1] / "png/1"
    lines = json.loads((fixture / "truth.json").read_text(encoding="utf-8"))["pages"][0]["lines"]
    regions = _source_regions(lines)
    regions[0]["bbox"][2] = 391
    _prediction(folder, regions)
    report = _tool().compare(fixture, **runs)
    assert not report["passed"]
    assert any("repeat " in s and "min_region_box_iou" in s for s in report["errors"])
    assert report["runs"][1]["pages"][0]["min_region_box_iou"] > 0.90


@pytest.mark.parametrize(
    "field",
    [
        "absent",
        "page_number",
        "width",
        "height",
        "schema_version",
        "coordinate_system",
        "label",
        "bbox",
        "bbox_pixels",
        "crop",
        "text",
        "region_count",
    ],
)
def test_native_grounding_manifest_must_match_actual_raw_page_and_crop(captures, field):
    fixture, runs = captures
    path = runs["native"][0] / "pdf/2/grounding.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if field == "absent":
        path.unlink()
    else:
        if field in ("page_number", "width", "height", "schema_version"):
            data[field] += 1
        elif field == "coordinate_system":
            data[field] = "pixels"
        elif field == "label":
            data["regions"][1]["label"] = "image"
        elif field in ("bbox", "bbox_pixels"):
            data["regions"][0][field][1] += 3
        elif field == "crop":
            data["regions"][3]["crop"] = "unrelated.jpg"
        elif field == "region_count":
            data["regions"].pop()
        else:
            data["regions"][0]["text"] = "不相关内容"
        _write_json(path, data)
    report = _tool().compare(fixture, **runs)
    assert not report["passed"]
    assert any("native-1/pdf/2" in s and "grounding.json" in s for s in report["errors"]), report["errors"]


@pytest.mark.parametrize(
    ("field", "mode"),
    [
        ("prompt_sha256", "mismatch"),
        ("weights_sha256", "mismatch"),
        ("prompt_sha256", "missing"),
        ("weights_sha256", "missing"),
        ("prompt_sha256", "invalid"),
        ("revision", "within_backend"),
        ("revision", "different_repos"),
    ],
)
def test_manifest_provenance_preserves_prompt_and_weights_but_allows_distinct_repos(captures, field, mode):
    fixture, runs = captures
    for backend, group in runs.items():
        for i, run in enumerate(group):
            file = run / "run.json"
            meta = json.loads(file.read_text(encoding="utf-8"))
            meta.update(prompt_sha256="a" * 64, weights_sha256="b" * 64, revision=("1" if backend == "native" else "2") * 40)
            if backend == "native" and i == 1:
                if mode == "mismatch":
                    meta[field] = "f" * 64
                elif mode == "missing":
                    del meta[field]
                elif mode == "invalid":
                    meta[field] = "not-a-hash"
                elif mode == "within_backend":
                    meta[field] = "3" * 40
            _write_json(file, meta)
    report = _tool().compare(fixture, **runs)
    if mode == "different_repos":
        assert report["passed"], report["errors"]
        assert report["comparable_inputs"] == {"prompt_sha256": "a" * 64, "weights_sha256": "b" * 64}
    else:
        assert not report["passed"] and any(field in s for s in report["errors"])


@pytest.fixture
def upgrade_captures(captures):
    fixture, runs = captures
    runs["reference_native"] = [shutil.copytree(p, p.with_name("reference-" + p.name)) for p in runs["native"]]
    fixture_hash = hashlib.sha256((fixture / "truth.json").read_bytes()).hexdigest()
    for group, version in (("legacy", "4.46.3"), ("native", "5.23.0"), ("reference_native", "5.17.0")):
        for run in runs[group]:
            path = run / "run.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            meta.update(
                transformers=version,
                torch="2.10.0+cu128",
                gpu="NVIDIA GeForce RTX 4090",
                fixture_sha256=fixture_hash,
                prompt_sha256="a" * 64,
                weights_sha256="b" * 64,
                revision=("1" if group == "legacy" else "2") * 40,
            )
            _write_json(path, meta)
    return fixture, runs


def _upgrade_report(fixture, runs, native_version="5.23.0", reference_native_version="5.17.0"):
    return _tool().compare(fixture, **runs, native_version=native_version, reference_native_version=reference_native_version)


def _change_title_width(runs, right):
    for run in runs:
        for path in run.rglob("grounding.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            title = data["regions"][0]
            title["bbox"][2], title["bbox_pixels"][2] = right, right * 200 // 999
            _prediction(path.parent, data["regions"])
            _write_json(path, data)


def test_upgrade_cli_compares_three_real_groups_without_grading_version_drift_as_repeat(upgrade_captures, tmp_path):
    fixture, runs = upgrade_captures
    _change_title_width(runs["native"], 391)
    out = tmp_path / "upgrade.json"
    args = [
        "compare",
        "--fixture",
        str(fixture),
        "--out",
        str(out),
        "--native-version",
        "5.23.0",
        "--reference-native-version",
        "5.17.0",
    ]
    for group, paths in runs.items():
        for path in paths:
            args.extend(["--" + group.replace("_", "-"), str(path)])
    assert _tool().main(args) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["passed"] and report["summary"]["measured_pages"] == 24
    assert report["expected_transformers"] == {"legacy": "4.46.3", "native": "5.23.0", "reference-native": "5.17.0"}
    assert report["common_runtime"] == {"torch": "2.10.0+cu128", "gpu": "NVIDIA GeForce RTX 4090"}
    assert {(r["group"], r["backend"], r["transformers"]) for r in report["runs"]} == {
        ("legacy", "legacy", "4.46.3"),
        ("reference-native", "native", "5.17.0"),
        ("native", "native", "5.23.0"),
    }
    assert all(r["torch"] == "2.10.0+cu128" and r["gpu"] == "NVIDIA GeForce RTX 4090" for r in report["runs"])
    assert all(p["native_grounding_verified"] for r in report["runs"] if r["backend"] == "native" for p in r["pages"])
    pairs = report["comparisons"]
    assert Counter(p["kind"] for p in pairs) == {"backends": 32, "native_versions": 16, "formats": 12, "repeat": 12}
    assert {p["right"].split("/")[0] for p in pairs if p["kind"] == "backends"} == {
        "reference-native-1",
        "reference-native-2",
        "native-1",
        "native-2",
    }
    for pair in (p for p in pairs if p["kind"] == "native_versions"):
        assert pair["left"].startswith("reference-native-") and pair["right"].startswith("native-")
        assert pair["native_accuracy_delta"] == 0 and 0.85 < pair["min_region_box_iou"] < 0.98


def test_upgrade_rejects_drift_between_natives_even_when_both_pass_against_legacy(upgrade_captures):
    fixture, runs = upgrade_captures
    _change_title_width(runs["reference_native"], 351)
    _change_title_width(runs["native"], 413)
    report = _upgrade_report(fixture, runs)
    assert not report["passed"] and report["errors"]
    assert all("native_versions " in error and "min_region_box_iou" in error for error in report["errors"])


@pytest.mark.parametrize(("worse_group", "regression"), [("native", True), ("reference_native", False)])
def test_upgrade_accuracy_delta_grades_candidate_relative_to_reference(upgrade_captures, worse_group, regression):
    fixture, runs = upgrade_captures
    for run in runs[worse_group]:
        for path in run.rglob("result.md"):
            text = path.read_text(encoding="utf-8")
            path.write_text(text[:-1], encoding="utf-8")
    report = _upgrade_report(fixture, runs)
    pairs = [p for p in report["comparisons"] if p["kind"] == "native_versions"]
    assert len(pairs) == 16
    assert all((p["native_accuracy_delta"] < -0.03 if regression else p["native_accuracy_delta"] > 0.03) for p in pairs)
    assert any("native_versions " in e and "native_accuracy_delta" in e for e in report["errors"]) is regression


@pytest.mark.parametrize(
    ("group", "field", "value"),
    [
        ("native", "transformers", "5.17.0"),
        ("reference_native", "transformers", None),
        ("legacy", "transformers", "4.47.0"),
        ("reference_native", "backend", "legacy"),
        ("reference_native", "revision", "3" * 40),
        ("all", "revision", "not-a-commit"),
        ("all", "revision", None),
        ("reference_native", "torch", "2.8.0+cu128"),
        ("native", "gpu", "NVIDIA T4"),
        ("all", "gpu", None),
        ("all", "fixture_sha256", None),
        ("native", "fixture_sha256", "d" * 64),
        ("all", "prompt_sha256", None),
        ("all", "weights_sha256", "not-a-hash"),
        ("reference_native", "weights_sha256", "c" * 64),
    ],
)
def test_upgrade_fails_closed_on_wrong_or_missing_provenance(upgrade_captures, group, field, value):
    fixture, runs = upgrade_captures
    if group == "all":
        paths = [p for group_paths in runs.values() for p in group_paths]
    elif field in {"revision", "torch", "gpu"}:
        paths = runs[group]  # Internally repeatable does not mean comparable to the other groups.
    else:
        paths = runs[group][:1]
    for path in paths:
        manifest = path / "run.json"
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if value is None:
            del meta[field]
        else:
            meta[field] = value
        _write_json(manifest, meta)
    report = _upgrade_report(fixture, runs)
    assert not report["passed"] and any(field in e for e in report["errors"]), report["errors"]


@pytest.mark.parametrize(("candidate", "reference"), [(None, "5.17.0"), ("5.23.0", None), ("5.*", "5.17.0"), ("5.17.0", "5.17.0")])
def test_upgrade_requires_distinct_explicit_exact_versions(upgrade_captures, candidate, reference):
    fixture, runs = upgrade_captures
    for group, version in (("native", candidate), ("reference_native", reference)):
        for run in runs[group]:
            path = run / "run.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            meta["transformers"] = version
            _write_json(path, meta)
    report = _upgrade_report(fixture, runs, candidate, reference)
    assert not report["passed"] and any("exact" in e or "distinct" in e for e in report["errors"])


def test_upgrade_requires_two_reference_runs_and_their_native_grounding(upgrade_captures):
    fixture, runs = upgrade_captures
    runs["reference_native"].pop()
    (runs["reference_native"][0] / "pdf/2/grounding.json").unlink()
    report = _upgrade_report(fixture, runs)
    assert not report["passed"]
    assert any("two distinct reference-native" in e for e in report["errors"])
    assert any("reference-native-1/pdf/2" in e and "grounding.json" in e for e in report["errors"])


def test_expected_versions_without_reference_runs_never_fall_back_to_two_way(upgrade_captures):
    fixture, runs = upgrade_captures
    del runs["reference_native"]
    report = _upgrade_report(fixture, runs)
    assert not report["passed"] and any("two distinct reference-native" in e for e in report["errors"])


@pytest.mark.parametrize("scope", ["same_run", "other_group", "same_run_directory", "shared_raw"])
def test_upgrade_rejects_reused_runs_or_pages(upgrade_captures, scope):
    fixture, runs = upgrade_captures
    candidate = runs["native"][0]
    path = candidate / "run.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    if scope == "same_run_directory":
        runs["reference_native"][0] = candidate
    elif scope == "same_run":
        meta["cases"]["pdf"][0] = meta["cases"]["png"][0]
    elif scope == "shared_raw":
        raw = candidate / "png/1/raw.txt"
        raw.unlink()
        try:
            raw.hardlink_to(runs["reference_native"][0] / "png/1/raw.txt")
        except (OSError, NotImplementedError):
            pytest.skip("hard links are not available on this filesystem")
    else:
        runs["reference_native"][0] = shutil.copytree(runs["reference_native"][0], candidate / "ref")
        meta["cases"]["png"][0] = "ref/png/1"
    _write_json(path, meta)
    report = _upgrade_report(fixture, runs)
    assert not report["passed"] and any("distinct" in e for e in report["errors"]), report["errors"]
