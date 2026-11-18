"""Record one real CUDA run for tools/deepseek_ocr2_compare.py in a fresh process."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import tomllib
from contextlib import ExitStack
from pathlib import Path
from time import monotonic
from unittest.mock import patch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WEIGHTS = "d8ff67a424ba6f4dd077885eb9d6a05d2537e76fe5491f0e2a9b712f8c8870fa"
# SHA-256 of the sorted {filename: Git blob ID} JSON from the pinned Hub revisions.
SNAPSHOTS = {
    "legacy": ("aaa02f3811945a91062062994c5c4a3f4c0af2b0",
               "d40b0278372428d776545af09a1ee9ad3ca5ed460499ee98346b81f3022bcb70", "model-00001-of-000001.safetensors"),
    "native": ("838dc37567aa97106fcd590e2be6a8f279881813",
               "53c8de45da930a9781c533efa80a27e2f017bd0b62355054a59b757cc3150b12", "model.safetensors"),
}
PAGE_SPLIT = "\n<--- Page Split --->\n"


def verify_snapshot(directory: Path, backend: str) -> None:
    _, expected, weights = SNAPSHOTS[backend]
    files = {}
    for path in directory.iterdir():
        if path.suffix in {".py", ".json"}:
            data = path.read_bytes()
            files[path.name] = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        elif path.name not in {weights, ".cache", ".gitattributes", "LICENSE.txt", "README.md", "assets"}:
            raise ValueError(f"Unexpected file in comparison snapshot: {path.name}")
    actual = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual != expected:
        raise ValueError(f"Not the pinned {backend} snapshot (model code or configuration differs)")
    with (directory / weights).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != WEIGHTS:
            raise ValueError("Checkpoint does not match the shared upstream weight SHA-256")


def pdf_output_pages(directory: Path, returned: str, count: int) -> list[Path]:
    from module.providers.ocr.document import rebase_generated_images

    pages = [directory / f"page_{i:04d}" for i in range(1, count + 1)]
    if set(directory.glob("page_*")) != set(pages):
        raise ValueError("Unexpected PDF page count")
    expected = PAGE_SPLIT.join(
        rebase_generated_images((page / "result.md").read_text(encoding="utf-8").strip(), page, directory)
        for page in pages
    )
    published = (directory / "result.md").read_text(encoding="utf-8").strip()
    if expected != returned.strip() or published != expected:
        raise ValueError("PDF page order, links or published document differ from the returned document")
    return pages


def fixture_digest(directory: Path) -> str:
    raw = (directory / "truth.json").read_bytes()
    truth = json.loads(raw)
    files = {p["file"]: p["sha256"] for p in truth["pages"]}
    files[truth["pdf"]] = truth["pdf_sha256"]
    if set(files) != {"page-1.png", "page-2.png", "pages.pdf"}:
        raise ValueError("Unexpected comparison fixture inputs")
    for name, expected in files.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Comparison fixture changed: {name}")
    return hashlib.sha256(raw).hexdigest()


def observe_legacy(stack: ExitStack, raw: list[str]) -> None:
    from utils.transformer_loader import transformerLoader

    load = transformerLoader.get_or_load_model
    observed = set()

    def get_model(loader, *args, **kwargs):
        model = load(loader, *args, **kwargs)
        namespace = model.infer.__func__.__globals__
        if id(namespace) not in observed:
            parse = namespace["re_match"]

            def record(decoded):
                raw.append(decoded)
                return parse(decoded)

            # Capture the exact decoded text passed to the upstream publisher,
            # not a second generation. Only literal coordinates are needed here.
            stack.enter_context(patch.dict(namespace, {"re_match": record, "eval": ast.literal_eval}))
            observed.add(id(namespace))
        return model

    stack.enter_context(patch.object(transformerLoader, "get_or_load_model", get_model))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=SNAPSHOTS, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="A new directory for one run")
    parser.add_argument("--prompt-toml", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "prompts.toml",
                        help="Read the exact deepseek_ocr_prompt from this project prompt file")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from rich.console import Console

    backend, fixture, model_dir, out = args.backend, args.fixture.resolve(), args.model_dir.resolve(), args.out.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the legacy/native acceptance comparison")
    if backend == "legacy" and transformers.__version__ != "4.46.3":
        raise RuntimeError("Use the isolated Transformers 4.46.3 legacy environment")
    if backend == "native" and transformers.__version__.split(".")[0] != "5":
        raise RuntimeError("Use the isolated Transformers 5.x native environment")
    prompt = tomllib.loads(args.prompt_toml.read_text(encoding="utf-8"))["prompts"]["deepseek_ocr_prompt"]
    if not isinstance(prompt, str) or prompt.count("<image>") != 1:
        raise ValueError("The comparison requires the same explicit single-image prompt for both backends")
    before = fixture_digest(fixture)
    verify_snapshot(model_dir, backend)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(17)
    started = monotonic()
    cases = {"png": [], "pdf": []}
    with ExitStack() as stack:
        raw: list[str] = []
        if backend == "legacy":
            from module.providers.ocr.deepseek import attempt_deepseek_ocr as attempt

            observe_legacy(stack, raw)
        else:
            from module.providers.ocr.deepseek_native import attempt_deepseek_native as attempt
        for name, category, rel in [("page-1.png", "png", "png/1"), ("page-2.png", "png", "png/2"), ("pages.pdf", "pdf", "pdf")]:
            target = out / rel
            raw.clear()
            print(f"Running {backend}: {name}", flush=True)
            result = attempt(uri=str(fixture / name), output_dir=str(target), model_id=str(model_dir),
                             console=Console(), progress=None, task_id=None,
                             prompt_text=prompt)
            pages = [target] if category == "png" else pdf_output_pages(target, result, 2)
            if backend == "legacy":
                if len(raw) != len(pages):
                    raise ValueError(f"Missing original legacy prediction for {name}")
                for page, decoded in zip(pages, raw):
                    with (page / "raw.txt").open("x", encoding="utf-8") as handle:
                        handle.write(decoded)
            cases[category].extend(p.relative_to(out).as_posix() for p in pages)
    if fixture_digest(fixture) != before:
        raise ValueError("Comparison fixture changed during inference")
    (out / "run.json").write_text(json.dumps({
        "backend": backend, "cases": cases, "revision": SNAPSHOTS[backend][0], "weights_sha256": WEIGHTS,
        "fixture_sha256": before,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "transformers": transformers.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "elapsed_seconds": round(monotonic() - started, 2),
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
