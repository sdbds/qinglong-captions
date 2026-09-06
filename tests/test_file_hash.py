import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.compat


def test_see_through_execution_plan_works_without_python311_file_digest(tmp_path, monkeypatch):
    from module.see_through import runner

    source = tmp_path / "image.png"
    source.write_bytes(b"abc")
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    config = SimpleNamespace(input_dir=tmp_path, skip_completed=True, save_to_psd=True)
    plan = runner.build_execution_plan(config, tmp_path / "out", [source])
    assert plan[0].source_fingerprint == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert plan[0].resume_stage == "layerdiff"


@pytest.mark.parametrize("body, expected", [
    (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
])
def test_sha256_file_matches_known_digests(tmp_path, body, expected):
    from utils.file_hash import sha256_file

    source = tmp_path / "input.bin"
    source.write_bytes(body)
    assert sha256_file(source) == expected
    assert sha256_file(str(source)) == expected


def test_sha256_file_reads_bounded_chunks(tmp_path, monkeypatch):
    from utils.file_hash import sha256_file

    body = b"x" * (2 * 1024 * 1024 + 7)
    source = tmp_path / "large.bin"
    source.write_bytes(body)
    original_open = Path.open
    read_sizes = []

    class BoundedReader:
        def __enter__(self):
            self.stream = original_open(source, "rb")
            return self

        def __exit__(self, *_):
            self.stream.close()

        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            read_sizes.append(size)
            return self.stream.read(size)

    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: BoundedReader())
    assert sha256_file(source) == hashlib.sha256(body).hexdigest()
    assert len(read_sizes) >= 3
