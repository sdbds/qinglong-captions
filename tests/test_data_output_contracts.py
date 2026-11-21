from types import SimpleNamespace

import numpy as np
import pytest


def test_exported_page_deduplicates_whole_links_without_touching_examples(tmp_path):
    from module.lanceexport import save_caption_by_pages

    header = '<header style="background-color: #f5f5f5;"><strong> Page 1 </strong></header>\n'
    sample = '![example](sub/fig(1)-front.png "sample")'
    managed = r".qinglong-ocr-artifacts/run/page/figure\(1\).png"
    examples = f"````md\n```\n{sample}\n\n\n![example]({managed})\n````\n\nInline `{sample}`.\n\n"
    source = (
        f'{examples}![front [view]](sub/fig(1)-front.png "Front view")\n'
        "![back](sub/fig(1)-back.png)\n"
        'Before ![again](<sub/fig(1)-front.png> "Another title") after.\n'
        f"![actual]({managed})\n"
    )

    assert save_caption_by_pages(tmp_path / "doc.md", [header + source], rebase_ocr_images=True)
    actual = (tmp_path / "doc" / "doc_1.md").read_text(encoding="utf-8")
    assert actual == (
        f'\n{examples}![front [view]](fig(1)-front.png "Front view")\n'
        "![back](fig(1)-back.png)\n"
        "Before  after.\n"
        f"![actual](../{managed})\n"
    )


def test_page_does_not_collapse_distinct_folders_or_modify_nonlocal_paths(tmp_path):
    from module.lanceexport import save_caption_by_pages

    header = '<header style="background-color: #f5f5f5;"><strong> Page 1 </strong></header>\n'
    body = (
        "![one](maps/icon.png)\n![two](charts/icon.png)\n"
        "![relative](../outside.png)\n![remote](https://example.com/part(1)/chart.png)\n"
    )
    assert save_caption_by_pages(tmp_path / "doc.md", [header + body])
    assert (tmp_path / "doc" / "doc_1.md").read_text(encoding="utf-8") == "\n" + body


def test_export_planning_does_not_claim_a_code_examples_managed_files(tmp_path):
    from utils.ocr_export import plan_ocr_image_assets

    url = ".qinglong-ocr-artifacts/0123456789abcdef-abcdefgh/page/figure.png"
    examples = f'Example `![sample]({url})`.\n\n~~~html\n<img src="{url}">\n~~~\n'
    assert plan_ocr_image_assets(examples, tmp_path / "src.pdf", tmp_path / "out.md", tmp_path / "out.pdf") == set()


@pytest.mark.parametrize("ending", ["\r\n", "\r"])
def test_image_source_ranges_and_page_edits_preserve_original_newlines(ending):
    from utils.markdown_images import iter_image_links
    from utils.ocr_export import rewrite_page_images

    example = "![sample](sub/fig(1).png)"
    live = ending.join(("![real", 'view](<sub/fig(1).png> "First view")'))
    duplicate = '![other](sub/fig(1).png "Second view")'
    prefix = ending.join(("Intro", "", "    " + example, "", "```md", example, "```", "", "Text: "))
    original = prefix + live + "!" + ending + duplicate + ending + "End.\n"

    links = list(iter_image_links(original))
    assert [(original[item.start : item.end], original[item.url_start : item.url_end]) for item in links] == [
        (live, "sub/fig(1).png"),
        (duplicate, "sub/fig(1).png"),
    ]
    assert rewrite_page_images(original, rebase=False) == prefix + live.replace("sub/", "") + "!" + ending * 2 + "End.\n"


def test_published_page_does_not_translate_existing_crlf_twice(tmp_path):
    from module.lanceexport import save_caption_by_pages

    header = '<header style="background-color: #f5f5f5;"><strong> Page 1 </strong></header>'
    body = "\r\nFirst\r\n\r\n![real](sub/figure.png)\r\nLast\n"
    assert save_caption_by_pages(tmp_path / "doc.md", [header + body])
    assert (tmp_path / "doc" / "doc_1.md").read_bytes() == body.replace("sub/", "").encode()


def test_html_source_ranges_ignore_comments_and_src_text_inside_other_attributes():
    from utils.markdown_images import iter_image_links

    block_tag = '<IMG alt="temperature > 20" data-src="lazy.png" SRC=figure.png>'
    inline_tag = '<img alt="literal src=\'wrong.png\' here" src="A&amp;B.png">'
    html = f'<div>\r\n<!-- draft: <img src="missing.png"> -->\r\n<![draft]>\r\n{block_tag}\r\n</div>\r\n\r\nText {inline_tag} end.'
    html += '\r\n\r\n<!-- draft --> <img src="last.png">'
    links = list(iter_image_links(html))
    assert [(html[item.start : item.end], html[item.url_start : item.url_end], item.href) for item in links] == [
        (block_tag, "figure.png", "figure.png"),
        (inline_tag, "A&amp;B.png", "A&B.png"),
        ('<img src="last.png">', "last.png", "last.png"),
    ]


def test_paginated_html_rebases_only_the_owned_src_outside_comments(tmp_path):
    from module.lanceexport import save_caption_by_pages
    from utils.ocr_artifacts import reserve_ocr_artifact_dir
    from utils.ocr_export import plan_ocr_image_assets

    source, output = tmp_path / "scan.pdf", tmp_path / "out"
    figure = reserve_ocr_artifact_dir(source) / "a&b.png"
    figure.write_bytes(b"figure")
    actual = figure.relative_to(tmp_path).as_posix().replace("&", "&amp;")
    missing = ".qinglong-ocr-artifacts/0123456789abcdef-abcdefgh/no.png"
    html = f'<div>\r\n<!-- <img src="{missing}"> -->\r\n<img alt="temperature > 20" src="{actual}">\r\n</div>'
    header = '<header style="background-color: #f5f5f5;"><strong> Page 1 </strong></header>\r\n\r\n'
    assets = plan_ocr_image_assets(header + html, source, output / "scan.md", output / "scan.pdf")
    assert {asset.source for asset in assets} == {figure}
    assert save_caption_by_pages(output / "scan.md", [header + html], rebase_ocr_images=True)
    published = (output / "scan" / "scan_1.md").read_bytes().decode()
    assert published == "\r\n\r\n" + html.replace(actual, "../" + actual)


@pytest.mark.parametrize(
    ("source", "limits", "expected"),
    [
        ((2048, 4), {"max_pixels": 8192}, (512, 16)),
        ((4, 2048), {"max_pixels": 8192}, (16, 512)),
        ((53, 53), {"max_long_edge": 45}, (32, 32)),
        ((400, 200), {"max_short_edge": 60}, (96, 48)),
        ((300, 300), {"max_pixels": 8000}, (80, 80)),
        ((2000, 1000), {"max_short_edge": 512}, (1024, 512)),
        ((4, 4), {}, (16, 16)),
    ],
)
def test_grid_aligned_resize_obeys_every_budget(source, limits, expected):
    from utils.stream_util import calculate_dimensions

    assert calculate_dimensions(*source, **({"max_long_edge": None} | limits)) == expected


@pytest.mark.parametrize("limits", [{"max_pixels": 255}, {"max_short_edge": 12}, {"max_long_edge": 15}, {"img_scale_num": 0}])
def test_impossible_resize_budget_is_reported(limits):
    from utils.stream_util import calculate_dimensions

    with pytest.raises(ValueError):
        calculate_dimensions(96, 96, **limits)


@pytest.mark.parametrize(
    ("repo", "output", "expected"),
    [
        ("cella110n/cl_tagger", [0.3, -2.0], [0.5744425, 0.1192029]),
        ("SmilingWolf/wd-v1-4-moat-tagger-v2", [0.3, 1.0], [0.3, 1.0]),
    ],
)
def test_public_tagger_load_and_infer_returns_probabilities(repo, output, expected, monkeypatch, tmp_path):
    from utils import wdtagger

    model = tmp_path / repo.replace("/", "_")
    if repo == "cella110n/cl_tagger":
        model /= "cl_tagger_1_02"
        model.mkdir(parents=True)
        (model / "tag_mapping.json").write_text('{"0":{"tag":"sample","category":"general"}}', encoding="utf-8")
    else:
        model.mkdir(parents=True)
        (model / "selected_tags.csv").write_text("tag_id,name,category\n0,sample,0\n", encoding="utf-8")
    (model / "model.onnx").touch()

    class Session:
        def run(self, _outputs, feeds):
            assert set(feeds) == {"input_1"}
            return [np.array([output], dtype=np.float32)]

    bundle = SimpleNamespace(session=Session(), providers=(), input_metas=[SimpleNamespace(name="input_1")])
    monkeypatch.setattr(wdtagger, "load_single_model_bundle", lambda **kwargs: bundle)
    args = SimpleNamespace(repo_id=repo, model_dir=str(tmp_path), force_download=False, remove_parents_tag=False)
    session, model_input, *_ = wdtagger.load_model_and_tags(args)
    assert model_input == "input_1"  # Also usable as a plain ONNX feed key.
    probs = wdtagger.process_batch([np.zeros((4, 4, 3), dtype=np.float32)], session, model_input)
    np.testing.assert_allclose(probs[0], expected, atol=1e-6)
