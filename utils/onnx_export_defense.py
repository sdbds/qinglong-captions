"""Export a defense reader checkpoint (model.py + model.safetensors) as a single ONNX graph."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

import numpy as np
import torch
from filelock import FileLock
from safetensors.torch import load_file

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module.music_export.service import publish_staged_outputs
from utils.file_hash import sha256_file

DEFAULT_MODEL_DIR = Path("datasets") / "model"
SOURCE_PACKAGE = "defense_model_source"
INPUT_NAMES = ["tiles", "candidates", "events", "prefix", "event_valid", "unknown_red"]
EXPORT_FORMAT = "qinglong-defense-onnx-v1"


def build_defense_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a defense reader checkpoint to ONNX (dynamic batch, candidate and event axes).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Trusted local directory containing model.py, heads.py, relations.py, metadata.json and the checkpoint. "
        "The exporter executes Python code from this directory; use only trusted sources.",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Explicit path to the .safetensors checkpoint.")
    parser.add_argument("--output-path", type=Path, default=None, help="Path to the output ONNX model.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an unchanged export from this checkpoint only.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for checkpoint load and ONNX tracing. Default: auto.",
    )
    parser.add_argument("--opset-version", type=int, default=20, help="ONNX opset version. Default: 20.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run onnx.checker and compare onnxruntime outputs against PyTorch on unseen input sizes.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Resolve assets and print the export plan without writing an ONNX file.",
    )
    return parser


def determine_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def import_model_source(model_dir: Path):
    # model.py uses relative imports; different source roots must never share the same module cache.
    directory = str(model_dir.resolve())
    name = f"{SOURCE_PACKAGE}_{hashlib.sha256(directory.encode('utf-8')).hexdigest()}"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [directory]
        sys.modules[name] = package
    return importlib.import_module(f"{name}.model")


def load_defense_model(model_dir: Path, checkpoint_path: Path, metadata: Mapping[str, Any]) -> torch.nn.Module:
    source = import_model_source(model_dir)
    architecture = metadata["architecture"]
    model = source.DefenseModel(architecture, head_schema=metadata["head_schema"])
    if not getattr(model.tiles, "uses_red", False):
        raise ValueError(f"Unsupported architecture for this exporter: {architecture}")
    model.load_state_dict(load_file(str(checkpoint_path)), strict=True)
    return model.eval()


class DefenseOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, output_names: list[str]):
        super().__init__()
        self.model = model
        self.output_names = output_names

    def forward(self, tiles, candidates, events, prefix, event_valid, unknown_red):
        output = self.model.tiles(tiles, candidates, events, prefix, event_valid, unknown_red)
        return tuple(output[name] for name in self.output_names)


def create_dummy_inputs(
    model: torch.nn.Module, device: torch.device, *, batch: int, candidates: int, events: int, seed: int
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    tile_dim = model.tiles.input1.in_features - 1
    binary = lambda *shape: (torch.rand(shape, generator=generator) > 0.7).float()
    event_valid = (torch.arange(events)[None, :] < torch.randint(1, events + 1, (batch, 1), generator=generator)).float()
    inputs = (
        torch.rand((batch, 3, 34, tile_dim), generator=generator),
        binary(batch, candidates, 36),
        binary(batch, events, 99),
        torch.rand((batch, 519), generator=generator),
        event_valid,
        binary(batch, 3).bool(),
    )
    return tuple(tensor.to(device) for tensor in inputs)


def export_to_onnx(wrapper: DefenseOnnxWrapper, output_path: Path, device: torch.device, opset_version: int) -> Path:
    dummy_inputs = create_dummy_inputs(wrapper.model, device, batch=2, candidates=4, events=16, seed=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch = torch.export.Dim("batch_size")
    candidates = torch.export.Dim("num_candidates")
    events = torch.export.Dim("num_events")

    with torch.inference_mode():
        program = torch.onnx.export(
            wrapper,
            dummy_inputs,
            opset_version=opset_version,
            dynamo=True,
            verbose=False,
            input_names=INPUT_NAMES,
            output_names=wrapper.output_names,
            dynamic_shapes={
                "tiles": {0: batch},
                "candidates": {0: batch, 1: candidates},
                "events": {0: batch, 1: events},
                "prefix": {0: batch},
                "event_valid": {0: batch, 1: events},
                "unknown_red": {0: batch},
            },
        )
    program.save(str(output_path))
    return output_path


def verify_export(wrapper: DefenseOnnxWrapper, output_path: Path, device: torch.device, *, atol: float = 1e-3) -> dict[str, float]:
    try:
        import onnx
        import onnxruntime
    except ImportError as exc:
        raise RuntimeError("The --verify flag requires the 'onnx' and 'onnxruntime' packages to be installed.") from exc

    onnx.checker.check_model(str(output_path))
    session = onnxruntime.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    report: dict[str, float] = {}
    # Sizes differ from the tracing sample so frozen shapes would be caught.
    for seed, (batch, candidates, events) in enumerate(((1, 1, 1), (3, 14, 61), (5, 7, 120)), start=1):
        inputs = create_dummy_inputs(wrapper.model, device, batch=batch, candidates=candidates, events=events, seed=seed)
        with torch.inference_mode():
            reference = wrapper(*inputs)
        actual = session.run(None, {name: tensor.cpu().numpy() for name, tensor in zip(INPUT_NAMES, inputs)})
        for name, expected, got in zip(wrapper.output_names, reference, actual):
            expected = expected.float().cpu().numpy()
            if expected.shape != got.shape:
                raise ValueError(f"{name} shape mismatch: {expected.shape} != {got.shape}")
            if not np.isfinite(expected).all() or not np.isfinite(got).all():
                raise ValueError(f"{name} contains non-finite values")
            report[name] = max(report.get(name, 0.0), float(np.abs(expected - got).max()))
    failed = {name: error for name, error in report.items() if error > atol}
    if failed:
        raise ValueError(f"ONNX outputs differ from PyTorch beyond {atol}: {failed}")
    return report


def build_export_metadata(
    wrapper: DefenseOnnxWrapper, metadata: Mapping[str, Any], checkpoint_path: Path, output_path: Path
) -> dict[str, Any]:
    tile_dim = wrapper.model.tiles.input1.in_features - 1
    return {
        "architecture": metadata["architecture"],
        "architecture_id": metadata.get("architecture_id"),
        "head_schema": metadata["head_schema"],
        "checkpoint_id": metadata.get("checkpoint_id"),
        "checkpoint_path": str(checkpoint_path),
        "onnx_path": str(output_path),
        "inputs": {
            "tiles": ["batch", 3, 34, tile_dim],
            "candidates": ["batch", "num_candidates", 36],
            "events": ["batch", "num_events", 99],
            "prefix": ["batch", 519],
            "event_valid": ["batch", "num_events"],
            "unknown_red": ["batch", 3],
        },
        "input_dtypes": {name: "bool" if name == "unknown_red" else "float32" for name in INPUT_NAMES},
        "output_names": wrapper.output_names,
        "export_notes": {
            "outputs_are_uncalibrated_logits": True,
        },
    }


def _metadata_digest(metadata: dict) -> str:
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _publication_lock(marker: Path) -> Path:
    # The JSON marker is shared by different casing of the graph suffix even on case-sensitive filesystems.
    lock = marker.with_name(f".{marker.name}.defense-export.lock")
    if lock.is_symlink() or (lock.exists() and (not lock.is_file() or lock.stat().st_size)):
        raise FileExistsError(f"Refusing to use an unrelated file as the export lock: {lock}")
    return lock


def _export_paths(output: Path, model_dir: Path, checkpoint: Path) -> tuple[Path, Path, Path]:
    if output.suffix.lower() != ".onnx":
        raise ValueError("The export path must have a .onnx suffix")
    graph = output.parent.resolve() / output.name
    paths = (graph, graph.with_suffix(".json"), graph.with_name(graph.name + ".data"))
    inputs = {checkpoint} | {(model_dir / name).resolve() for name in ("model.py", "heads.py", "relations.py", "metadata.json")}
    if any(path.resolve() in inputs for path in (*paths, _publication_lock(paths[1]))):
        raise ValueError("An export path would overwrite a model input")
    return paths


def _previous_export(paths: tuple[Path, Path, Path], model_dir: Path, checkpoint: Path, overwrite: bool) -> set[str]:
    graph, marker, data = paths
    if any(path.resolve() != path or path.is_symlink() or (path.exists() and not path.is_file()) for path in paths):
        raise FileExistsError(f"An export destination is redirected or is not a regular file: {graph}")
    existing = {path.name for path in paths if path.exists()}
    if not existing:
        return set()
    if not overwrite:
        raise FileExistsError(f"Export destination already exists: {graph}; choose a new name or --overwrite an owned export")
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError) as exc:
        raise FileExistsError(f"No valid export ownership metadata at {marker}") from exc
    signature = metadata.pop("metadata_sha256", None) if isinstance(metadata, dict) else None
    if not signature or signature != _metadata_digest(metadata):
        raise FileExistsError(f"Unrecognized or edited export metadata: {marker}")
    files = metadata.get("artifact_sha256")
    if not (
        metadata.get("export_format") == EXPORT_FORMAT and metadata.get("model_dir") == str(model_dir)
        and metadata.get("checkpoint_path") == str(checkpoint) and metadata.get("onnx_path") == str(graph)
        and isinstance(files, dict) and {graph.name} <= files.keys() <= {graph.name, data.name}
        and files.keys() == existing - {marker.name}
        and all(sha256_file(graph.parent / name) == digest for name, digest in files.items())
    ):
        raise FileExistsError(f"Refusing to replace foreign, missing or edited export artifacts: {graph}")
    return set(files)


def run_defense_export(args: argparse.Namespace) -> int:
    model_dir = args.model_dir.resolve()
    checkpoint_path = (args.checkpoint_path or model_dir / "model.safetensors").resolve()
    paths = _export_paths(args.output_path or checkpoint_path.with_suffix(".onnx"), model_dir, checkpoint_path)
    output_path, metadata_path, data_path = paths
    device = determine_device(args.device)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

    plan = {
        "model_dir": str(model_dir),
        "checkpoint_path": str(checkpoint_path),
        "output_path": str(output_path),
        "device": str(device),
        "architecture": metadata["architecture"],
        "head_schema": metadata["head_schema"],
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)

    if args.print_only:
        return 0

    _previous_export(paths, model_dir, checkpoint_path, args.overwrite)
    model = load_defense_model(model_dir, checkpoint_path, metadata).to(device)
    with torch.inference_mode():
        sample = model.tiles(*create_dummy_inputs(model, device, batch=1, candidates=1, events=1, seed=0))
    wrapper = DefenseOnnxWrapper(model, list(sample)).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".defense-onnx-", dir=output_path.parent) as temporary:
        stage = Path(temporary)
        graph = stage / output_path.name
        export_to_onnx(wrapper, graph, device, args.opset_version)
        artifacts = list(stage.iterdir())
        if not graph.is_file() or any(
            item.name not in {graph.name, data_path.name} or item.is_symlink() or not item.is_file() or not item.stat().st_size
            for item in artifacts
        ):
            raise ValueError("Exporter produced an incomplete graph or an unsupported external-data layout")
        if args.verify:
            report = verify_export(wrapper, graph, device)
            print(f"Verified against PyTorch (max abs logit error): {json.dumps(report, indent=2)}", flush=True)
        else:
            import onnx

            onnx.checker.check_model(str(graph))
        result = build_export_metadata(wrapper, metadata, checkpoint_path, output_path)
        digests = {item.name: sha256_file(item) for item in artifacts}
        result.update(export_format=EXPORT_FORMAT, model_dir=str(model_dir), artifact_sha256=digests)
        result["metadata_sha256"] = _metadata_digest(result)
        (stage / metadata_path.name).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        lock = _publication_lock(metadata_path)
        with FileLock(lock, timeout=0, preserve_lock_file=True):
            if lock.stat().st_size:
                raise FileExistsError(f"Refusing to use an unrelated file as the export lock: {lock}")
            previous = _previous_export(paths, model_dir, checkpoint_path, args.overwrite)
            publish_staged_outputs(
                stage, output_path.parent, managed_names=previous | digests.keys(), metadata_name=metadata_path.name,
            )
    print(f"Exported ONNX model: {output_path}", flush=True)
    print(f"Export metadata: {metadata_path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_defense_export(build_defense_arg_parser().parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[onnx_export_defense] {exc}", file=sys.stderr, flush=True)
        raise
