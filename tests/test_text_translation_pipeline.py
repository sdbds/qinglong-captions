import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import lance
import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'module'))

from module.lanceImport import load_data, transform2lance
from module.lanceexport import save_caption
from module.texttranslate import (
    TranslationRuntimeError,
    load_or_create_dataset,
    merge_translations,
    normalize_dataset,
    protect_markdown,
    preserve_chunk_whitespace,
    resolve_translated_markdown_path,
    restore_placeholders,
    translate_dataset,
)
from utils.doc_normalize import normalize_text_asset
from utils.lance_blob import take_blob_files
from utils.text_chunker import compute_chunk_offsets, slice_by_offsets


def test_load_data_keeps_sidecars_and_standalone_text_by_default(tmp_path):
    media = tmp_path / 'clip.mp4'
    media.write_bytes(b'video')
    (tmp_path / 'clip.txt').write_text('paired caption', encoding='utf-8')
    (tmp_path / 'notes.txt').write_text('standalone text', encoding='utf-8')

    default_rows = load_data(str(tmp_path))
    assert [Path(row['file_path']).name for row in default_rows] == ['clip.mp4', 'notes.txt']
    assert default_rows[0]['caption'] == ['paired caption']
    assert default_rows[1]['caption'] == []

    media_only_rows = load_data(str(tmp_path), include_text_assets=False)
    assert [Path(row['file_path']).name for row in media_only_rows] == ['clip.mp4']


def test_normalize_txt_to_markdown():
    markdown = normalize_text_asset(Path('sample.txt'), b'Hello\n\nWorld')
    assert markdown == 'Hello\n\nWorld\n'


def test_chunk_offsets_roundtrip():
    markdown = (
        '# Title\n\n'
        '> quoted line one. quoted line two.\n>\n> quoted line three.\n\n'
        '- first item. second sentence.\n'
        '- second item.\n\n'
        '| col |\n| --- |\n| value |\n\n'
        '```python\nprint(1)\n```\n\n'
        'Last line.\n'
    )
    offsets = compute_chunk_offsets(markdown, max_chars=30)
    chunks = slice_by_offsets(markdown, offsets)
    assert ''.join(chunks) == markdown
    assert offsets[-1] == len(markdown)
    assert len(offsets) >= 2


def test_save_caption_with_language_suffix(tmp_path):
    base = tmp_path / 'foo.md'
    ok = save_caption(str(base), ['# Hello\n'], 'text', caption_suffix='_zh_cn', caption_extension='.md')
    assert ok is True
    assert (tmp_path / 'foo_zh_cn.md').read_text(encoding='utf-8') == '# Hello\n'


def test_save_caption_uses_structured_caption_extension_from_json(tmp_path):
    base = tmp_path / 'song.wav'
    ok = save_caption(
        str(base),
        ['{"description":"A short song summary","caption_extension":".txt"}'],
        'audio',
    )

    assert ok is True
    assert (tmp_path / 'song.txt').read_text(encoding='utf-8') == 'A short song summary'
    assert (tmp_path / 'song.json').read_text(encoding='utf-8').strip()


class UppercaseTranslator:
    def translate(self, text, source_lang, target_lang, *, context='', glossary=''):
        return text.upper()


def test_normalize_and_translate_dataset_roundtrip(tmp_path):
    source_file = tmp_path / 'story.txt'
    source_file.write_text('hello world.\n\nnext line.', encoding='utf-8')
    source_bytes = source_file.read_bytes()

    dataset = transform2lance(str(tmp_path), output_name='sample', save_binary=True, tag='raw.import.test')
    assert dataset is not None

    dataset_path = tmp_path / 'sample.lance'
    raw_ds = lance.dataset(str(dataset_path), version='raw.import.test')
    raw_row = raw_ds.to_table().to_pylist()[0]
    assert raw_row['captions'] == []
    assert raw_row['chunk_offsets'] == []
    assert take_blob_files(raw_ds, [0], 'blob')[0].readall() == source_bytes

    normalize_dataset(dataset_path, source_version='raw.import.test', norm_tag='norm.docling.test', max_chars=10)
    norm_ds = lance.dataset(str(dataset_path), version='norm.docling.test')
    norm_row = norm_ds.to_table().to_pylist()[0]
    assert norm_row['captions'] == ['hello world.\n\nnext line.\n']
    assert norm_row['chunk_offsets'][-1] == len(norm_row['captions'][0])
    assert take_blob_files(norm_ds, [0], 'blob')[0].readall() == source_bytes

    translate_dataset(
        dataset_path=dataset_path,
        source_version='norm.docling.test',
        translation_tag='tr.mock.zh_cn.test',
        translator=UppercaseTranslator(),
        source_lang='en',
        target_lang='zh_cn',
        max_chars=10,
        context_chars=0,
        glossary='',
    )
    tr_ds = lance.dataset(str(dataset_path), version='tr.mock.zh_cn.test')
    tr_row = tr_ds.to_table().to_pylist()[0]
    assert tr_row['captions'] == ['HELLO WORLD.\n\nNEXT LINE.\n']
    assert tr_row['chunk_offsets'][-1] == len(tr_row['captions'][0])
    assert take_blob_files(tr_ds, [0], 'blob')[0].readall() == source_bytes


def test_load_or_create_dataset_reuses_existing_lance(tmp_path):
    existing = tmp_path / 'existing.lance'
    existing.mkdir()

    dataset_path, created = load_or_create_dataset(str(tmp_path), output_name='sample', raw_tag='raw.test')

    assert dataset_path == existing
    assert created is False


def test_load_or_create_dataset_prefers_matching_output_name(tmp_path):
    (tmp_path / 'aaa.lance').mkdir()
    matching = tmp_path / 'sample.lance'
    matching.mkdir()

    dataset_path, created = load_or_create_dataset(str(tmp_path), output_name='sample', raw_tag='raw.test')

    assert dataset_path == matching
    assert created is False


def test_load_or_create_dataset_raises_on_ambiguous_existing_lance_dirs(tmp_path):
    (tmp_path / 'aaa.lance').mkdir()
    (tmp_path / 'bbb.lance').mkdir()

    with pytest.raises(TranslationRuntimeError):
        load_or_create_dataset(str(tmp_path), output_name='sample', raw_tag='raw.test')


def test_load_or_create_dataset_force_reimport_calls_transform(tmp_path):
    with patch('module.texttranslate.transform2lance', return_value=object()) as mock_transform:
        dataset_path, created = load_or_create_dataset(
            str(tmp_path),
            output_name='sample',
            raw_tag='raw.test',
            force_reimport=True,
        )

    assert dataset_path == tmp_path / 'sample.lance'
    assert created is True
    mock_transform.assert_called_once_with(
        dataset_dir=str(tmp_path),
        output_name='sample',
        save_binary=True,
        not_save_disk=False,
        tag='raw.test',
        include_text_assets=True,
    )


def test_merge_translations_reads_saved_markdown_and_updates_tag(tmp_path):
    translated = tmp_path / 'story_zh_cn.md'
    translated.write_text('translated body\n', encoding='utf-8')

    schema = pa.schema(
        [
            pa.field('uris', pa.string()),
            pa.field('captions', pa.list_(pa.string())),
            pa.field('chunk_offsets', pa.list_(pa.int32())),
        ]
    )

    executed_tables = []

    class MergeBuilder:
        def when_matched_update_all(self):
            return self

        def execute(self, table):
            executed_tables.append(table)

    class TargetDataset:
        def __init__(self):
            self.schema = schema

        def merge_insert(self, on):
            assert on == 'uris'
            return MergeBuilder()

    latest_dataset = SimpleNamespace(
        version=4,
        tags=SimpleNamespace(create=MagicMock(), update=MagicMock()),
    )

    with patch('module.texttranslate.lance.dataset', side_effect=[TargetDataset(), latest_dataset]):
        merged = merge_translations(
            dataset_path=tmp_path / 'sample.lance',
            base_version='norm.test',
            translation_tag='tr.test',
            merge_candidates=['story.txt'],
            current_run_translations={},
            export_root=tmp_path,
            target_lang='zh-cn',
            max_chars=8,
            merge_batch_size=1,
        )

    assert merged == 1
    assert len(executed_tables) == 1
    row = executed_tables[0].to_pylist()[0]
    assert row['uris'] == 'story.txt'
    assert row['captions'] == ['translated body\n']
    assert row['chunk_offsets'][-1] == len('translated body\n')
    latest_dataset.tags.create.assert_called_once_with('tr.test', 4)


def test_translation_helpers_preserve_protected_tokens_and_whitespace():
    original = '  code: `x = 1` and https://example.com/docs  '
    masked, replacements = protect_markdown(original)

    assert masked != original
    restored = restore_placeholders(masked, replacements)
    assert restored == original
    assert preserve_chunk_whitespace('  hello \n', 'world') == '  world \n'


def test_resolve_translated_markdown_path_preserves_relative_directories(tmp_path):
    export_root = tmp_path / 'exports'

    first = resolve_translated_markdown_path(Path('a/readme.txt'), export_root, 'zh_cn')
    second = resolve_translated_markdown_path(Path('b/readme.txt'), export_root, 'zh_cn')

    assert first == export_root / 'a' / 'readme_zh_cn.md'
    assert second == export_root / 'b' / 'readme_zh_cn.md'
