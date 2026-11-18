"""Standalone offline ConvRot8 W8A8 quantization.

The rotation, packing, and CUDA/Triton inference path follow the ConvRot8
implementation in ostris/ai-toolkit's toolkit/util/convrot_quant.py.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterator, Mapping

import torch
import torch.nn.functional as F
from filelock import FileLock
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from module.music_export.service import publish_staged_outputs

CONVROT_BLOCK = 16
LAYERDIFF3D_REPO = "layerdifforg/seethroughv0.0.2_layerdiff3d"
LAYERDIFF3D_SUBFOLDER = "unet"
LAYERDIFF3D_MODEL_CLASS = "module.see_through.vendor.modules.layerdiffuse.layerdiff3d:UNetFrameConditionModel"


def largest_pow4_divisor(value: int) -> int:
    if value <= 0:
        raise ValueError("A power-of-four divisor requires a positive dimension")
    divisor = 1
    while value >= 4 and value % 4 == 0:
        value //= 4
        divisor *= 4
    return divisor


def rotation_size_for(in_features: int, max_rotation_size: int = 256) -> int:
    return min(max_rotation_size, largest_pow4_divisor(in_features))


def linear_is_convrot8_eligible(
    in_features: int,
    out_features: int,
    max_rotation_size: int = 256,
) -> bool:
    return (
        in_features > 0
        and out_features > 0
        and in_features % CONVROT_BLOCK == 0
        and out_features % 8 == 0
        and rotation_size_for(in_features, max_rotation_size) >= CONVROT_BLOCK
    )


@lru_cache(maxsize=None)
def _regular_hadamard_cpu(rot_size: int) -> torch.Tensor:
    r4 = torch.tensor(
        [[1.0, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    matrix = r4
    while matrix.shape[0] < rot_size:
        matrix = torch.kron(matrix, r4)
    if matrix.shape[0] != rot_size:
        raise ValueError(f"rot_size must be a power of 4, got {rot_size}")
    return matrix / rot_size**0.5


def regular_hadamard(
    rot_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return _regular_hadamard_cpu(rot_size).to(device=device, dtype=dtype)


def rotate(tensor: torch.Tensor, rot_size: int) -> torch.Tensor:
    if rot_size == 1:
        return tensor
    matrix = regular_hadamard(rot_size, tensor.device, tensor.dtype)
    shape = tensor.shape
    blocks = tensor.reshape(-1, shape[-1] // rot_size, rot_size)
    return torch.matmul(blocks, matrix).reshape(shape)


def quantize_int8_rows(tensor: torch.Tensor, qmax: int = 127) -> tuple[torch.Tensor, torch.Tensor]:
    values = tensor.float()
    scales = values.abs().amax(dim=1) / qmax
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    qdata = torch.round(values / scales[:, None]).clamp_(-qmax, qmax).to(torch.int8)
    return qdata, scales


_int8_kernels: tuple[Any, Any] | None = None


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _get_int8_kernels() -> tuple[Any, Any]:
    global _int8_kernels
    if _int8_kernels is not None:
        return _int8_kernels

    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    globals()["triton"] = triton
    globals()["tl"] = tl
    globals()["libdevice"] = libdevice

    @triton.jit
    def int8_act_quant_kernel(
        x_ptr,
        q_ptr,
        scales_ptr,
        width,
        QMAX: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        base = row * width
        max_values = tl.zeros((BLOCK_K,), tl.float32)
        for column_start in range(0, width, BLOCK_K):
            offsets = column_start + tl.arange(0, BLOCK_K)
            values = tl.load(x_ptr + base + offsets, mask=offsets < width, other=0.0).to(tl.float32)
            max_values = tl.maximum(max_values, tl.abs(values))
        absolute_max = tl.max(max_values, axis=0)
        scale = tl.where(absolute_max > 0, absolute_max / QMAX, 1.0)
        for column_start in range(0, width, BLOCK_K):
            offsets = column_start + tl.arange(0, BLOCK_K)
            mask = offsets < width
            values = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
            quantized = libdevice.rint(values / scale)
            quantized = tl.minimum(tl.maximum(quantized, -1.0 * QMAX), 1.0 * QMAX)
            tl.store(q_ptr + base + offsets, quantized.to(tl.int8), mask=mask)
        tl.store(scales_ptr + row, scale)

    @triton.jit
    def int8_epilogue_kernel(
        accum_ptr,
        activation_scales_ptr,
        weight_scales_ptr,
        bias_ptr,
        output_ptr,
        width,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        column_block = tl.program_id(1)
        offsets = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < width
        accum = tl.load(accum_ptr + row * width + offsets, mask=mask, other=0).to(tl.float32)
        activation_scale = tl.load(activation_scales_ptr + row)
        weight_scale = tl.load(weight_scales_ptr + offsets, mask=mask, other=0.0)
        output = accum * (activation_scale * weight_scale)
        if HAS_BIAS:
            output += tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(output_ptr + row * width + offsets, output.to(output_ptr.dtype.element_ty), mask=mask)

    _int8_kernels = int8_act_quant_kernel, int8_epilogue_kernel
    return _int8_kernels


def quantize_int8_rows_fused(
    tensor: torch.Tensor,
    qmax: int = 127,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = tensor.shape
    padded_rows = ((rows + 31) // 32) * 32
    if tensor.is_cuda and _triton_available():
        quantized = torch.empty((padded_rows, width), dtype=torch.int8, device=tensor.device)
        scales = torch.empty(padded_rows, dtype=torch.float32, device=tensor.device)
        if padded_rows != rows:
            quantized[rows:].zero_()
            scales[rows:].fill_(1.0)
        activation_kernel, _ = _get_int8_kernels()
        block_k = min(2048, 1 << (width - 1).bit_length())
        activation_kernel[(rows,)](
            tensor.contiguous(),
            quantized,
            scales,
            width,
            QMAX=qmax,
            BLOCK_K=block_k,
            num_warps=8,
        )
        return quantized, scales

    quantized, scales = quantize_int8_rows(tensor, qmax)
    if padded_rows != rows:
        quantized = F.pad(quantized, (0, 0, 0, padded_rows - rows))
        scales = F.pad(scales, (0, padded_rows - rows), value=1.0)
    return quantized, scales


def _int8_epilogue(
    accum: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if accum.is_cuda and _triton_available():
        rows, width = accum.shape
        output = torch.empty((rows, width), dtype=output_dtype, device=accum.device)
        _, epilogue_kernel = _get_int8_kernels()
        epilogue_kernel[(rows, (width + 1023) // 1024)](
            accum,
            activation_scales,
            weight_scales,
            bias if bias is not None else activation_scales,
            output,
            width,
            HAS_BIAS=bias is not None,
            BLOCK_N=1024,
            num_warps=4,
        )
        return output

    output = accum.float() * activation_scales[:, None] * weight_scales[None, :]
    if bias is not None:
        output = output + bias.float()
    return output.to(output_dtype)


def quantize_weight(
    weight: torch.Tensor,
    max_rotation_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if weight.ndim != 2:
        raise ValueError(f"ConvRot8 weights must be rank 2, got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if not linear_is_convrot8_eligible(in_features, out_features, max_rotation_size):
        raise ValueError(
            "ConvRot8 requires in_features divisible by 16, out_features divisible by 8, "
            "and a power-of-4 rotation block of at least 16"
        )
    rot_size = rotation_size_for(in_features, max_rotation_size)
    qdata, scales = quantize_int8_rows(rotate(weight.float(), rot_size))
    return qdata, scales, rot_size


@lru_cache(maxsize=None)
def int8_mm_supported(device_type: str, device_index: int | None) -> bool:
    if device_type != "cuda" or not torch.cuda.is_available():
        return False
    device = torch.device(device_type, device_index)
    try:
        left = torch.zeros((32, 64), dtype=torch.int8, device=device)
        right = torch.zeros((64, 32), dtype=torch.int8, device=device)
        torch._int_mm(left, right)
    except (RuntimeError, NotImplementedError):
        return False
    return True


class ConvRotLinear(nn.Linear):
    is_convrot8_quantized = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        rot_size: int,
        original_dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.cr8_rot_size = int(rot_size)
        self.original_dtype = original_dtype
        self.last_backend = "not_run"
        self.register_buffer(
            "cr8_qdata",
            torch.empty((out_features, in_features), dtype=torch.int8, device=device),
        )
        self.register_buffer(
            "cr8_scales",
            torch.empty(out_features * 4, dtype=torch.uint8, device=device),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=original_dtype, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_quantized(
        cls,
        *,
        qdata: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
        rot_size: int,
        original_dtype: torch.dtype,
    ) -> "ConvRotLinear":
        layer = cls(
            qdata.shape[1],
            qdata.shape[0],
            bias=bias is not None,
            rot_size=rot_size,
            original_dtype=original_dtype,
            device=qdata.device,
        )
        layer.cr8_qdata.copy_(qdata)
        layer.cr8_scales.copy_(scales.contiguous().view(torch.uint8))
        if bias is not None:
            layer.bias.copy_(bias)
        return layer

    @property
    def scales(self) -> torch.Tensor:
        return self.cr8_scales.view(torch.float32)

    @property
    def weight(self) -> torch.Tensor:
        rotated_weight = self.cr8_qdata.float() * self.scales[:, None]
        return rotate(rotated_weight, self.cr8_rot_size).to(self.original_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rotated_x = rotate(x, self.cr8_rot_size)
        if int8_mm_supported(x.device.type, x.device.index):
            flat = rotated_x.reshape(-1, self.in_features)
            rows = flat.shape[0]
            activation_codes, activation_scales = quantize_int8_rows_fused(flat)
            accum = torch._int_mm(activation_codes, self.cr8_qdata.t())[:rows]
            output = _int8_epilogue(
                accum,
                activation_scales[:rows],
                self.scales,
                self.bias,
                x.dtype,
            )
            self.last_backend = "cuda_int8"
            return output.reshape(*x.shape[:-1], self.out_features)

        self.last_backend = "dequantized"
        rotated_weight = (self.cr8_qdata.float() * self.scales[:, None]).to(x.dtype)
        return F.linear(rotated_x, rotated_weight, self.bias)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _weight_key(module_name: str) -> str:
    return f"{module_name}.weight" if module_name else "weight"


def _state_key(module_name: str, suffix: str) -> str:
    return f"{module_name}.{suffix}" if module_name else suffix


class SafetensorSource:
    def __init__(self, source_dir: Path):
        self.source_dir = source_dir
        self.weight_map: dict[str, Path] = {}
        self.shape_map: dict[str, tuple[int, ...]] = {}
        self._keys_by_path: dict[Path, list[str]] = {}
        self._scan()

    def _scan(self) -> None:
        index_paths = sorted(self.source_dir.glob("*.safetensors.index.json"))
        if index_paths:
            if len(index_paths) != 1:
                raise ValueError(f"Expected one Safetensors index in {self.source_dir}, found {len(index_paths)}")
            index = json.loads(index_paths[0].read_text(encoding="utf-8"))
            self.weight_map = {key: (self.source_dir / filename).resolve() for key, filename in index["weight_map"].items()}
            for key, path in self.weight_map.items():
                self._keys_by_path.setdefault(path, []).append(key)
            for path, keys in self._keys_by_path.items():
                with safe_open(str(path), framework="pt", device="cpu") as reader:
                    available = set(reader.keys())
                    for key in keys:
                        if key not in available:
                            raise KeyError(f"Safetensors index points to missing key {key!r} in {path}")
                        self.shape_map[key] = tuple(reader.get_slice(key).get_shape())
            return

        tensor_paths = sorted(self.source_dir.glob("*.safetensors"))
        if not tensor_paths:
            raise FileNotFoundError(f"No .safetensors checkpoint found in {self.source_dir}")

        for tensor_path in tensor_paths:
            resolved_path = tensor_path.resolve()
            with safe_open(str(resolved_path), framework="pt", device="cpu") as reader:
                for key in reader.keys():
                    if key in self.weight_map:
                        raise ValueError(f"Duplicate tensor key {key!r} in {self.source_dir}")
                    self.weight_map[key] = resolved_path
                    self.shape_map[key] = tuple(reader.get_slice(key).get_shape())
                    self._keys_by_path.setdefault(resolved_path, []).append(key)

    def keys(self) -> list[str]:
        return sorted(self.weight_map)

    def iter_shards(self) -> Iterator[tuple[Path, list[str]]]:
        for path in sorted(self._keys_by_path, key=str):
            yield path, sorted(self._keys_by_path[path])

    def tensor_shape(self, key: str) -> tuple[int, ...]:
        return self.shape_map[key]


class _ShardWriter:
    def __init__(self, output_dir: Path, max_shard_bytes: int):
        self.output_dir = output_dir
        self.max_shard_bytes = max_shard_bytes
        self.pending: dict[str, torch.Tensor] = {}
        self.pending_bytes = 0
        self.shard_paths: list[Path] = []
        self.weight_map: dict[str, str] = {}
        self.total_tensor_bytes = 0

    def add(self, key: str, tensor: torch.Tensor) -> None:
        tensor = tensor.detach().cpu().contiguous()
        tensor_bytes = _tensor_nbytes(tensor)
        if self.pending and self.pending_bytes + tensor_bytes > self.max_shard_bytes:
            self._flush()
        self.pending[key] = tensor
        self.pending_bytes += tensor_bytes
        self.total_tensor_bytes += tensor_bytes

    def _flush(self) -> None:
        if not self.pending:
            return
        path = self.output_dir / f".convrot-shard-{len(self.shard_paths) + 1:05d}.safetensors"
        save_file(self.pending, path, metadata={"format": "pt", "quantization": "convrot8"})
        self.shard_paths.append(path)
        for key in self.pending:
            self.weight_map[key] = path.name
        self.pending = {}
        self.pending_bytes = 0

    def flush(self) -> None:
        self._flush()

    def finish(self) -> dict[str, Any]:
        self._flush()
        shard_count = len(self.shard_paths)
        renamed: dict[str, str] = {}
        for index, old_path in enumerate(self.shard_paths, start=1):
            new_name = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
            old_name = old_path.name
            old_path.rename(self.output_dir / new_name)
            renamed[old_name] = new_name
        weight_map = {key: renamed[name] for key, name in self.weight_map.items()}
        index = {
            "metadata": {"total_size": self.total_tensor_bytes, "quantization": "convrot8"},
            "weight_map": weight_map,
        }
        (self.output_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return index


def _quantize_weight_from_reader(
    reader: Any,
    key: str,
    *,
    in_features: int,
    out_features: int,
    rot_size: int,
    device: torch.device,
    chunk_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
    rows_per_chunk = max(1, chunk_bytes // max(in_features * 8, 1))
    q_chunks: list[torch.Tensor] = []
    scale_chunks: list[torch.Tensor] = []
    original_dtype: torch.dtype | None = None
    seen_rows = 0
    tensor_slice = reader.get_slice(key)

    for start in range(0, out_features, rows_per_chunk):
        source_chunk = tensor_slice[start : min(start + rows_per_chunk, out_features)]
        original_dtype = source_chunk.dtype
        weight_chunk = source_chunk.to(device=device, dtype=torch.float32)
        qdata, scales = quantize_int8_rows(rotate(weight_chunk, rot_size))
        q_chunks.append(qdata.cpu())
        scale_chunks.append(scales.cpu())
        seen_rows += source_chunk.shape[0]
        del weight_chunk, qdata, scales

    if seen_rows != out_features or original_dtype is None:
        raise ValueError(f"Unexpected shape while reading {key}: expected {out_features} rows, read {seen_rows}")
    return torch.cat(q_chunks), torch.cat(scale_chunks), original_dtype


def _bundle_files(directory: Path) -> set[str]:
    entries = list(directory.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("Bundle contains a link or a non-file entry")
    manifest = json.loads((directory / "convrot_config.json").read_text(encoding="utf-8"))
    index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
    if not (
        isinstance(manifest, dict) and manifest.get("quantization") == "convrot8" and manifest.get("format_version") == 1
    ):
        raise ValueError("Invalid ConvRot manifest")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map or not all(
        isinstance(name, str) and re.fullmatch(r"model-[0-9]{5,}-of-[0-9]{5,}\.safetensors", name)
        for name in weight_map.values()
    ):
        raise ValueError("Invalid ConvRot shard index")
    required = {"convrot_config.json", "model.safetensors.index.json", *weight_map.values()}
    names = {entry.name for entry in entries}
    if not required <= names <= required | {"config.json"} or any(entry.stat().st_size == 0 for entry in entries):
        raise ValueError("Bundle contains unowned, empty or missing files")
    return names


def _previous_bundle_files(output_dir: Path, overwrite: bool) -> set[str]:
    if output_dir.resolve() != output_dir:
        raise ValueError(f"Output directory redirects elsewhere: {output_dir}")
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return set()
    if not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    try:
        return _bundle_files(output_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise FileExistsError(
            f"Refusing to overwrite a directory that is not exclusively a ConvRot bundle: {output_dir}"
        ) from exc


@contextmanager
def _staged_bundle(source_dir: Path, output_dir: Path, overwrite: bool) -> Iterator[Path]:
    if source_dir.is_relative_to(output_dir) or output_dir.is_relative_to(source_dir):
        raise ValueError(f"Checkpoint and output directories must not overlap: {source_dir} and {output_dir}")
    _previous_bundle_files(output_dir, overwrite)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".convrot-stage-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        yield stage
        new_files = _bundle_files(stage)
        # Conversions can run concurrently, but ownership must be rechecked at publication.
        with FileLock(output_dir.parent / f".{output_dir.name}.convrot.lock", timeout=0):
            previous = _previous_bundle_files(output_dir, overwrite)
            if not output_dir.exists():
                stage.rename(output_dir)
            else:
                publish_staged_outputs(
                    stage, output_dir, managed_names=previous | new_files, metadata_name="convrot_config.json",
                )


def quantize_checkpoint(
    *,
    model: nn.Module,
    source_dir: Path | str,
    output_dir: Path | str,
    device: torch.device,
    max_rotation_size: int = 256,
    max_shard_bytes: int = 2 * 1024**3,
    chunk_bytes: int = 256 * 1024**2,
    include: str | None = None,
    exclude: str | None = None,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    source_dir = Path(source_dir).resolve()
    output_path = Path(output_dir)
    output_dir = output_path.resolve()
    if output_dir != output_path.parent.resolve() / output_path.name:
        raise ValueError(f"Output directory redirects elsewhere: {output_path}")
    with _staged_bundle(source_dir, output_dir, overwrite) as stage:
        return _write_checkpoint(
            model, source_dir, stage, device=device, max_rotation_size=max_rotation_size,
            max_shard_bytes=max_shard_bytes, chunk_bytes=chunk_bytes, include=include, exclude=exclude, progress=progress,
        )


def _write_checkpoint(
    model: nn.Module,
    source_dir: Path,
    output_dir: Path,
    *,
    device: torch.device,
    max_rotation_size: int,
    max_shard_bytes: int,
    chunk_bytes: int,
    include: str | None,
    exclude: str | None,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    source = SafetensorSource(source_dir)
    include_pattern = re.compile(include) if include else None
    exclude_pattern = re.compile(exclude) if exclude else None

    quantized_by_key: dict[str, tuple[str, nn.Linear, int]] = {}
    skipped_modules: dict[str, str] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, ConvRotLinear):
            continue
        if include_pattern and not include_pattern.search(module_name):
            skipped_modules[module_name] = "include_pattern"
            continue
        if exclude_pattern and exclude_pattern.search(module_name):
            skipped_modules[module_name] = "exclude_pattern"
            continue
        if not linear_is_convrot8_eligible(module.in_features, module.out_features, max_rotation_size):
            skipped_modules[module_name] = "shape"
            continue
        key = _weight_key(module_name)
        if key not in source.weight_map:
            skipped_modules[module_name] = "missing_weight"
            continue
        expected_shape = (module.out_features, module.in_features)
        if source.tensor_shape(key) != expected_shape:
            raise ValueError(
                f"Linear shape mismatch for {key}: model has {expected_shape}, checkpoint has {source.tensor_shape(key)}"
            )
        quantized_by_key[key] = (
            module_name,
            module,
            rotation_size_for(module.in_features, max_rotation_size),
        )

    writer = _ShardWriter(output_dir, max_shard_bytes)
    quantized_modules: dict[str, dict[str, Any]] = {}
    original_tensor_bytes = 0
    quantized_parameter_count = 0
    total_parameter_count = 0
    quantized_done = 0
    if progress:
        progress(f"Discovered {len(quantized_by_key)} eligible Linear weights and {len(source.weight_map)} checkpoint tensors.")

    for source_path, source_keys in source.iter_shards():
        with safe_open(str(source_path), framework="pt", device="cpu") as reader:
            for key in source_keys:
                if key not in quantized_by_key:
                    tensor = reader.get_tensor(key)
                    original_tensor_bytes += _tensor_nbytes(tensor)
                    total_parameter_count += tensor.numel()
                    writer.add(key, tensor)
                    continue

                module_name, module, rot_size = quantized_by_key[key]
                source_shape = source.tensor_shape(key)
                qdata, scales, original_dtype = _quantize_weight_from_reader(
                    reader,
                    key,
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rot_size=rot_size,
                    device=device,
                    chunk_bytes=chunk_bytes,
                )
                source_element_size = torch.empty((), dtype=original_dtype).element_size()
                original_tensor_bytes += source_shape[0] * source_shape[1] * source_element_size
                total_parameter_count += source_shape[0] * source_shape[1]
                writer.add(_state_key(module_name, "cr8_qdata"), qdata)
                writer.add(_state_key(module_name, "cr8_scales"), scales.contiguous().view(torch.uint8))
                quantized_parameter_count += qdata.numel()
                quantized_modules[module_name] = {
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "rot_size": rot_size,
                    "original_dtype": str(original_dtype).removeprefix("torch."),
                    "bias": module.bias is not None,
                    "source_weight_key": key,
                }
                quantized_done += 1
                if progress and (quantized_done == 1 or quantized_done % 25 == 0 or quantized_done == len(quantized_by_key)):
                    progress(
                        f"Quantized {quantized_done}/{len(quantized_by_key)}: {module_name} "
                        f"({module.out_features}x{module.in_features}, rotation={rot_size})"
                    )
        writer.flush()

    index = writer.finish()
    config_path = source_dir / "config.json"
    if config_path.is_file():
        shutil.copy2(config_path, output_dir / "config.json")

    manifest: dict[str, Any] = {
        "format_version": 1,
        "quantization": "convrot8",
        "source_dir": str(source_dir),
        "max_rotation_size": max_rotation_size,
        "quantized_modules": quantized_modules,
        "skipped_linear_modules": skipped_modules,
        "quantized_parameter_count": quantized_parameter_count,
        "total_parameter_count": total_parameter_count,
        "quantized_parameter_fraction": (quantized_parameter_count / total_parameter_count if total_parameter_count else 0.0),
        "original_tensor_bytes": original_tensor_bytes,
        "quantized_tensor_bytes": index["metadata"]["total_size"],
    }
    (output_dir / "convrot_config.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def summarize_manifest(manifest: Mapping[str, Any], output_dir: Path | str) -> dict[str, Any]:
    original_bytes = int(manifest["original_tensor_bytes"])
    quantized_bytes = int(manifest["quantized_tensor_bytes"])
    return {
        "output_dir": str(Path(output_dir).resolve()),
        "quantized_modules": len(manifest["quantized_modules"]),
        "skipped_linear_modules": len(manifest["skipped_linear_modules"]),
        "quantized_parameter_count": int(manifest["quantized_parameter_count"]),
        "total_parameter_count": int(manifest["total_parameter_count"]),
        "quantized_parameter_fraction": float(manifest["quantized_parameter_fraction"]),
        "original_tensor_bytes": original_bytes,
        "quantized_tensor_bytes": quantized_bytes,
        "tensor_byte_reduction": 1.0 - quantized_bytes / original_bytes if original_bytes else 0.0,
    }


def _replace_module(root: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)


def install_convrot_modules(model: nn.Module, manifest: Mapping[str, Any]) -> None:
    for module_name, spec in manifest["quantized_modules"].items():
        original = model.get_submodule(module_name)
        if not isinstance(original, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {module_name}, got {type(original).__name__}")
        original_dtype = getattr(torch, spec["original_dtype"])
        replacement = ConvRotLinear(
            spec["in_features"],
            spec["out_features"],
            bias=spec["bias"],
            rot_size=spec["rot_size"],
            original_dtype=original_dtype,
            device=original.weight.device,
        )
        _replace_module(model, module_name, replacement)


def load_quantized_checkpoint(
    *,
    model: nn.Module,
    bundle_dir: Path | str,
    device: torch.device,
) -> nn.Module:
    bundle_dir = Path(bundle_dir).resolve()
    manifest = json.loads((bundle_dir / "convrot_config.json").read_text(encoding="utf-8"))
    index = json.loads((bundle_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    install_convrot_modules(model, manifest)

    loaded_keys: set[str] = set()
    shard_names = sorted(set(index["weight_map"].values()))
    for shard_name in shard_names:
        state = load_file(bundle_dir / shard_name, device=str(device))
        model.load_state_dict(state, strict=False, assign=True)
        loaded_keys.update(state)
        del state

    expected_keys = set(model.state_dict())
    missing_keys = sorted(expected_keys - loaded_keys)
    unexpected_keys = sorted(loaded_keys - expected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(f"Quantized checkpoint state mismatch: missing={missing_keys[:8]}, unexpected={unexpected_keys[:8]}")
    return model


def parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)?\s*", value, flags=re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid byte size: {value!r}")
    number = float(match.group(1))
    unit = (match.group(2) or "").lower()
    multipliers = {
        "": 1,
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }
    return int(number * multipliers[unit])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline streaming ConvRot8 W8A8 quantizer for Safetensors checkpoints. "
            "The output is a custom bundle loaded with load_quantized_checkpoint()."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    quantize_parser = subparsers.add_parser("quantize", help="Quantize eligible nn.Linear weights.")
    quantize_parser.add_argument("--profile", choices=("layerdiff3d",), default=None)
    quantize_parser.add_argument("--model", default=None, help="Hugging Face repo ID or local model directory.")
    quantize_parser.add_argument("--subfolder", default=None, help="Component subfolder containing config and weights.")
    quantize_parser.add_argument(
        "--model-class",
        default=None,
        help="Import path in module:Class form; the class must implement load_config/from_config.",
    )
    quantize_parser.add_argument("--revision", default=None)
    quantize_parser.add_argument("--output-dir", type=Path, required=True)
    quantize_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    quantize_parser.add_argument("--max-rotation-size", type=int, default=256)
    quantize_parser.add_argument("--max-shard-size", dest="max_shard_bytes", type=parse_size, default=parse_size("2GiB"))
    quantize_parser.add_argument("--chunk-size", dest="chunk_bytes", type=parse_size, default=parse_size("256MiB"))
    quantize_parser.add_argument("--include", default=None, help="Regex selecting module names.")
    quantize_parser.add_argument("--exclude", default=None, help="Regex excluding module names.")
    quantize_parser.add_argument("--local-files-only", action="store_true")
    quantize_parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    if args.command == "quantize" and args.profile == "layerdiff3d":
        args.model = args.model or LAYERDIFF3D_REPO
        args.subfolder = args.subfolder or LAYERDIFF3D_SUBFOLDER
        args.model_class = args.model_class or LAYERDIFF3D_MODEL_CLASS
    if args.command == "quantize":
        if not args.model:
            raise ValueError("--model is required when no profile supplies it")
        if not args.model_class:
            raise ValueError("--model-class is required when no profile supplies it")
    return args


def determine_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_source_dir(
    model: str,
    *,
    subfolder: str | None,
    revision: str | None,
    local_files_only: bool,
) -> Path:
    local_path = Path(model).expanduser()
    if local_path.exists():
        source_dir = local_path / subfolder if subfolder else local_path
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Model component directory does not exist: {source_dir}")
        return source_dir.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Remote model IDs require huggingface_hub") from exc

    prefix = f"{subfolder.strip('/')}/" if subfolder else ""
    snapshot_path = snapshot_download(
        repo_id=model,
        revision=revision,
        allow_patterns=[
            f"{prefix}*.json",
            f"{prefix}*.safetensors",
            f"{prefix}*.safetensors.index.json",
        ],
        local_files_only=local_files_only,
    )
    source_dir = Path(snapshot_path) / subfolder if subfolder else Path(snapshot_path)
    return source_dir.resolve()


def import_model_class(spec: str) -> type[nn.Module]:
    module_name, separator, attribute_path = spec.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError(f"Model class must use module:Class form, got {spec!r}")
    value: Any = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    if not isinstance(value, type) or not issubclass(value, nn.Module):
        raise TypeError(f"{spec!r} does not resolve to a torch.nn.Module class")
    return value


def instantiate_model_skeleton(model_class: type[nn.Module], config_path: Path) -> nn.Module:
    try:
        from accelerate import init_empty_weights
    except ImportError as exc:
        raise RuntimeError("Model skeleton creation requires accelerate") from exc

    config = model_class.load_config(config_path)
    with init_empty_weights():
        model = model_class.from_config(config)
    return model


def run_quantize(args: argparse.Namespace) -> int:
    device = determine_device(args.device)
    source_dir = resolve_source_dir(
        args.model,
        subfolder=args.subfolder,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    config_path = source_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config does not exist: {config_path}")
    model_class = import_model_class(args.model_class)
    model = instantiate_model_skeleton(model_class, config_path)

    plan = {
        "model": args.model,
        "source_dir": str(source_dir),
        "output_dir": str(args.output_dir.resolve()),
        "model_class": args.model_class,
        "device": str(device),
        "max_rotation_size": args.max_rotation_size,
        "max_shard_bytes": args.max_shard_bytes,
        "chunk_bytes": args.chunk_bytes,
    }
    print(json.dumps(plan, indent=2), flush=True)
    manifest = quantize_checkpoint(
        model=model,
        source_dir=source_dir,
        output_dir=args.output_dir,
        device=device,
        max_rotation_size=args.max_rotation_size,
        max_shard_bytes=args.max_shard_bytes,
        chunk_bytes=args.chunk_bytes,
        include=args.include,
        exclude=args.exclude,
        overwrite=args.overwrite,
        progress=lambda message: print(f"[convrot_quant] {message}", flush=True),
    )
    print(json.dumps(summarize_manifest(manifest, args.output_dir), indent=2), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "quantize":
        return run_quantize(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[convrot_quant] {exc}", file=sys.stderr, flush=True)
        raise
