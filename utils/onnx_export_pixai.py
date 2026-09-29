"""Export the official PixAI tagger as a single FP32 ONNX graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from filelock import FileLock
from PIL import Image

from module.music_export import atomic_output_path
from module.wdtagger.pixai import PIXAI_SOURCE_REPO_ID, preprocess_pixai_images


def build_pixai_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export PixAI Tagger v1.0 to FP32 ONNX (dynamic batch, fixed image size).")
    parser.add_argument("--model-dir", type=Path, default=Path("wd14_tagger_model") / PIXAI_SOURCE_REPO_ID.replace("/", "_"))
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--download", action="store_true", help="Download the official weights and custom code at the selected revision."
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing export only after the new export succeeds.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--opset-version", type=int, default=20)
    parser.add_argument(
        "--verify", action="store_true", help="Check the graph, original-model parity, preprocessing, and ONNX batches 1 and 2."
    )
    parser.add_argument(
        "--verify-image", action="append", type=Path, default=[], help="Also verify this local image; may be repeated."
    )
    parser.add_argument("--print-only", action="store_true")
    return parser


def _real_rotary(self, q, k):
    def rotate(x):
        pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        real, imag = pairs[..., 0], pairs[..., 1]
        result = torch.stack((real * self.rope_cos - imag * self.rope_sin, real * self.rope_sin + imag * self.rope_cos), dim=-1)
        return result.flatten(-2).type_as(x)

    return rotate(q), rotate(k)


def prepare_pixai_for_onnx(model: torch.nn.Module) -> int:
    count = 0
    for module in model.modules():
        if not getattr(module, "use_rope", False):
            continue
        frequencies = getattr(module, "freqs_cis", None)
        if frequencies is None:
            continue
        module.register_buffer("rope_cos", frequencies.real.clone(), persistent=False)
        module.register_buffer("rope_sin", frequencies.imag.clone(), persistent=False)
        del module.freqs_cis
        module._apply_rope = MethodType(_real_rotary, module)
        count += 1
    return count


def compare_logits(reference: np.ndarray, actual: np.ndarray, *, probability_atol: float = 1e-4) -> dict:
    if reference.shape != actual.shape:
        raise ValueError(f"Logit shape mismatch: {reference.shape} != {actual.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("Verification requires finite logits")
    reference_probs = 1 / (1 + np.exp(-np.clip(reference.astype(np.float64), -80, 80)))
    actual_probs = 1 / (1 + np.exp(-np.clip(actual.astype(np.float64), -80, 80)))
    delta = np.abs(reference_probs - actual_probs)
    report = {
        "max_logit_error": float(np.max(np.abs(reference - actual))),
        "max_probability_error": float(delta.max()),
        "mean_probability_error": float(delta.mean()),
    }
    if report["max_probability_error"] > probability_atol:
        raise ValueError(f"ONNX probability error exceeds {probability_atol}: {report}")
    return report


def _verification_images(paths):
    rng = np.random.default_rng(20260929)
    images = [
        Image.fromarray(rng.integers(0, 256, (301, 503, 3), dtype=np.uint8)),
        Image.fromarray(rng.integers(0, 256, (457, 239, 4), dtype=np.uint8)),
    ]
    for path in paths:
        with Image.open(path) as image:
            images.append(image.copy())
    return images


def run_pixai_export(args) -> int:
    directory = args.model_dir.resolve()
    output = (args.output_path or directory / "model.onnx").resolve()
    if output.suffix.lower() != ".onnx":
        raise ValueError("ONNX output must use .onnx and must not overwrite source model metadata")
    plan = {
        "repo_id": PIXAI_SOURCE_REPO_ID,
        "revision": args.revision,
        "model_dir": str(directory),
        "output_path": str(output),
        "input_name": "pixel_values",
        "output_name": "logits",
        "dtype": "float32",
        "opset": args.opset_version,
        "dynamic_batch": True,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.print_only:
        return 0
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Export already exists: {output}; use --overwrite to replace it")
    if args.force_download and not args.download:
        raise ValueError("--force-download requires --download")
    if args.download:
        from huggingface_hub import snapshot_download

        snapshot_download(
            PIXAI_SOURCE_REPO_ID,
            revision=args.revision,
            local_dir=directory,
            force_download=args.force_download,
            allow_patterns=["config.json", "preprocessor_config.json", "tagger_pipeline.py", "model.safetensors", "README.md"],
        )

    from transformers import AutoImageProcessor, AutoModel
    from utils.onnx_export import determine_device

    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    size = int(config["img_size"])
    device = determine_device(args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"Loading official FP32 model on {device}...", flush=True)
    model = AutoModel.from_pretrained(directory, trust_remote_code=True, local_files_only=True).float().to(device).eval()
    processor = AutoImageProcessor.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
    cases = []
    if args.verify:
        images = _verification_images(args.verify_image)
        pixels = processor(images)["pixel_values"]
        np.testing.assert_array_equal(preprocess_pixai_images(images, size), pixels.numpy())
        batches = [pixels[:1], pixels[:2]] + [pixels[i : i + 1] for i in range(2, len(images))]
        with torch.inference_mode():
            for batch in batches:
                reference = model(batch.to(device)).float().cpu().numpy()
                cases.append((batch, reference))
        print("Captured original-model references; preprocessing matches exactly.", flush=True)

    patched = prepare_pixai_for_onnx(model)
    print(f"Replaced {patched} complex RoPE operations with real arithmetic.", flush=True)
    if patched != config["depth"]:
        raise ValueError(f"Unexpected PixAI RoPE layout: patched {patched}, expected {config['depth']}")
    with torch.inference_mode():
        for pixels, reference in cases:
            compare_logits(reference, model(pixels.to(device)).float().cpu().numpy())

    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"), atomic_output_path(output) as temporary:
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Another export already created {output}; use --overwrite to replace it")
        dummy = torch.zeros(1, 3, size, size, device=device)
        print("Exporting single-file ONNX...", flush=True)
        with torch.inference_mode():
            torch.onnx.export(
                model,
                (dummy,),
                str(temporary),
                export_params=True,
                opset_version=args.opset_version,
                dynamo=False,
                external_data=False,
                do_constant_folding=True,
                input_names=["pixel_values"],
                output_names=["logits"],
                dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
            )
        del model, dummy
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if args.verify:
            import onnx
            import onnxruntime as ort

            print("Checking ONNX graph and numerical parity...", flush=True)
            onnx.checker.check_model(str(temporary))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 8
            providers = (
                [("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"]
                if device.type == "cuda"
                else ["CPUExecutionProvider"]
            )
            if device.type == "cuda" and hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            session = ort.InferenceSession(str(temporary), sess_options=options, providers=providers)
            for pixels, reference in cases:
                actual = session.run(["logits"], {"pixel_values": pixels.numpy()})[0]
                report = {"batch_size": len(pixels), **compare_logits(reference, actual)}
                print(json.dumps(report), flush=True)
            del session
    print(f"Exported ONNX model: {output} ({output.stat().st_size:,} bytes)", flush=True)
    return 0
