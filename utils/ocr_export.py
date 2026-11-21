"""Copy image links generated in a project-owned OCR run without adopting user files."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from urllib.parse import unquote

from utils.markdown_images import iter_image_links
from utils.ocr_artifacts import (
    OCR_ARTIFACT_ROOT,
    ensure_ocr_artifact_root,
    ocr_artifact_run_prefix,
    require_ocr_artifact_root,
)

_LEGACY_PAGE_IMAGE = re.compile(r"(?!\.{1,2}/)[^/:?#\\]+/([^/\\?#]+)")
_RUN_NAME = re.compile(r"[a-f0-9]{16}-[a-z0-9_]{8}")
_PORTABLE_SOURCE = ".portable-source"


def _managed_parts(url: str) -> tuple[str, ...] | None:
    decoded = unquote(url.removeprefix("<").removesuffix(">"))
    parts = tuple(decoded.split("/"))
    if parts[0] != OCR_ARTIFACT_ROOT:
        return None
    if len(parts) < 3 or parts[2] == _PORTABLE_SOURCE or any(
        part in {"", ".", ".."} or any(char in part for char in ':\\\x00<>|?*')
        or part.rstrip(" .") != part or PureWindowsPath(part).is_reserved()
        for part in parts
    ):
        raise ValueError(f"Invalid OCR asset link: {url}")
    return parts


def rewrite_page_images(markdown: str, *, rebase: bool, normalize_legacy: bool = True) -> str:
    links = list(iter_image_links(markdown))
    flat_targets = {}
    if normalize_legacy:
        for link in links:
            if link.is_markdown:
                match = _LEGACY_PAGE_IMAGE.fullmatch(link.href)
                target = match[1] if match else link.href
                flat_targets.setdefault(target, set()).add(link.href)

    seen, pieces, cursor = set(), [], 0
    for link in links:
        url = markdown[link.url_start:link.url_end]
        if normalize_legacy and link.is_markdown and link.href in seen:
            pieces.append(markdown[cursor:link.start])
            cursor = link.end
            continue
        if link.is_markdown:
            seen.add(link.href)
        if rebase and _managed_parts(link.href) is not None:
            replacement = "../" + url
        elif normalize_legacy and link.is_markdown and (match := _LEGACY_PAGE_IMAGE.fullmatch(link.href)):
            # Flatten old one-folder layouts only when distinct images cannot collide.
            if len(flat_targets[match[1]]) != 1 or "/" not in url:
                continue
            replacement = url.split("/", 1)[1]
        else:
            continue
        pieces.extend((markdown[cursor:link.url_start], replacement))
        cursor = link.url_end
    pieces.append(markdown[cursor:])
    return "".join(pieces)


def rebase_ocr_page_images(markdown: str) -> str:
    return rewrite_page_images(markdown, rebase=True, normalize_legacy=False)


def _physical_entry(path: Path, *, directory: bool) -> os.stat_result | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    correct_type = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode) and info.st_nlink == 1
    if not correct_type or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"OCR asset path is not a physical {'directory' if directory else 'file'}: {path}")
    return info


@dataclass(frozen=True)
class OCRAsset:
    source_root: Path
    destination_root: Path
    relative: tuple[str, ...]
    source_name: str
    destination_name: str

    @property
    def source(self) -> Path:
        return self.source_root.joinpath(*self.relative)

    @property
    def destination(self) -> Path:
        return self.destination_root.joinpath(*self.relative)

    @property
    def destination_binding(self) -> Path:
        return self.destination_root / self.relative[0] / _PORTABLE_SOURCE

    def _binding(self, root: Path) -> str | None:
        run = root / self.relative[0]
        if _physical_entry(run, directory=True) is None:
            return None
        marker = run / _PORTABLE_SOURCE
        if _physical_entry(marker, directory=False) is None:
            return None
        return marker.read_text(encoding="utf-8")

    def verify_source(self) -> os.stat_result:
        if _physical_entry(self.source_root, directory=True) is not None:
            require_ocr_artifact_root(self.source_root)
            info = self._asset_entry(self.source_root)
            if info is not None:
                bound_name = self._binding(self.source_root)
                if bound_name is None:
                    belongs = self.relative[0].startswith(ocr_artifact_run_prefix(Path(self.source_name)))
                else:
                    belongs = os.path.normcase(bound_name) == os.path.normcase(self.source_name)
                if not belongs:
                    raise ValueError(f"OCR asset does not belong to the source document: {self.source}")
                return info
        raise FileNotFoundError(f"Missing OCR asset: {self.source}")

    def _asset_entry(self, root: Path) -> os.stat_result | None:
        path = root
        for part in self.relative[:-1]:
            path = path / part
            if _physical_entry(path, directory=True) is None:
                return None
        return _physical_entry(path / self.relative[-1], directory=False)

    def verify_destination(self) -> bool:
        if _physical_entry(self.destination_root, directory=True) is None:
            return False
        require_ocr_artifact_root(self.destination_root)
        bound_name = self._binding(self.destination_root)
        if (self.source != self.destination and bound_name is not None
                and os.path.normcase(bound_name) != os.path.normcase(self.destination_name)):
            raise ValueError(f"Conflicting OCR asset source binding: {self.destination_binding}")
        info = self._asset_entry(self.destination_root)
        if info is None:
            return False
        if self.source != self.destination:
            with self.source.open("rb") as source, self.destination.open("rb") as destination:
                if self.source.stat().st_size != info.st_size or (
                    hashlib.file_digest(source, "sha256").digest() != hashlib.file_digest(destination, "sha256").digest()
                ):
                    raise ValueError(f"Conflicting OCR asset at export destination: {self.destination}")
        return True


def plan_ocr_image_assets(markdown: str, source: Path, caption_target: Path, media_target: Path) -> set[OCRAsset]:
    assets = set()
    for link in iter_image_links(markdown):
        parts = _managed_parts(link.href)
        if parts is None:
            continue
        if not _RUN_NAME.fullmatch(parts[1]):
            raise ValueError(f"Invalid OCR asset run: {link.href}")
        # Markdown resolves beside the caption; a later Lance import resolves beside the media.
        for parent in {caption_target.parent, media_target.parent}:
            assets.add(OCRAsset(
                source.parent.resolve() / parts[0], parent / parts[0], parts[1:],
                source.name, media_target.name,
            ))
    for asset in sorted(assets, key=lambda item: item.destination):
        asset.verify_source()
        asset.verify_destination()
    return assets


@contextmanager
def publish_ocr_image_assets():
    created = {}

    def create_without_replacing(target, writer):
        fd, tmp_path = tempfile.mkstemp(prefix=".ocr-export-", dir=target.parent)
        temporary = Path(tmp_path)
        try:
            with os.fdopen(fd, "wb") as prepared:
                writer(prepared)
            published = temporary.stat()
            # Install only if the target is still free.
            try:
                if os.name == "nt":
                    temporary.rename(target)
                else:
                    os.link(temporary, target)
            except FileExistsError:
                return False
            created[target] = published
            return True
        finally:
            temporary.unlink(missing_ok=True)

    def publish(asset: OCRAsset) -> None:
        expected_source = asset.verify_source()
        existing = asset.verify_destination()
        if asset.source == asset.destination:
            return
        ensure_ocr_artifact_root(asset.destination_root)
        parent = asset.destination_root
        for part in asset.relative[:-1]:
            parent = parent / part
            parent.mkdir(exist_ok=True)
            _physical_entry(parent, directory=True)

        # Re-imports address the exported primary's name, not the original run prefix.
        if not asset.destination_binding.exists():
            create_without_replacing(
                asset.destination_binding, lambda marker: marker.write(asset.destination_name.encode("utf-8")),
            )
            existing = asset.verify_destination()
        if existing:
            return

        def copy_source(prepared):
            with asset.source.open("rb") as source:
                if not os.path.samestat(expected_source, os.fstat(source.fileno())):
                    raise ValueError(f"OCR asset changed while preparing export: {asset.source}")
                shutil.copyfileobj(source, prepared)

        if not create_without_replacing(asset.destination, copy_source) and not asset.verify_destination():
            raise FileExistsError(f"OCR asset appeared during export: {asset.destination}")

    try:
        yield publish
    except BaseException:
        for path, original in reversed(created.items()):
            try:
                if os.path.samestat(original, path.lstat()):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
