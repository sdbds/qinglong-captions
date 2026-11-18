import importlib
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn


def _convrot():
    try:
        return importlib.import_module("utils.convrot_quant")
    except ModuleNotFoundError:
        pytest.fail("utils.convrot_quant has not been implemented")


def test_regular_hadamard_rotation_is_self_inverse():
    convrot = _convrot()
    values = torch.arange(64, dtype=torch.float32).reshape(2, 32)

    rotated = convrot.rotate(values, rot_size=16)
    restored = convrot.rotate(rotated, rot_size=16)

    torch.testing.assert_close(restored, values)


def test_linear_eligibility_matches_convrot8_shape_contract():
    convrot = _convrot()

    assert convrot.linear_is_convrot8_eligible(1280, 1280)
    assert not convrot.linear_is_convrot8_eligible(1279, 1280)
    assert not convrot.linear_is_convrot8_eligible(1280, 1279)
    assert not convrot.linear_is_convrot8_eligible(8, 16)


def test_nonpositive_dimensions_have_no_convrot_factorization():
    convrot = _convrot()
    for value in (-1, 0):
        with pytest.raises(ValueError):
            convrot.largest_pow4_divisor(value)
    assert not convrot.linear_is_convrot8_eligible(0, 32)
    assert not convrot.linear_is_convrot8_eligible(64, 0)


def test_weight_quantization_uses_rotated_per_output_channel_int8():
    convrot = _convrot()
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(32, 64, generator=generator)

    qdata, scales, rot_size = convrot.quantize_weight(weight)
    reconstructed = convrot.rotate(qdata.float() * scales[:, None], rot_size)
    relative_l2 = torch.linalg.vector_norm(reconstructed - weight) / torch.linalg.vector_norm(weight)

    assert qdata.dtype == torch.int8
    assert scales.dtype == torch.float32
    assert rot_size == 64
    assert relative_l2.item() < 0.01


def test_convrot_linear_cpu_forward_uses_quantized_rotated_weight():
    convrot = _convrot()
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(32, 64, generator=generator)
    bias = torch.randn(32, generator=generator)
    x = torch.randn(3, 5, 64, generator=generator)
    qdata, scales, rot_size = convrot.quantize_weight(weight)
    layer = convrot.ConvRotLinear.from_quantized(
        qdata=qdata,
        scales=scales,
        bias=bias,
        rot_size=rot_size,
        original_dtype=weight.dtype,
    )

    actual = layer(x)
    expected = F.linear(convrot.rotate(x, rot_size), qdata.float() * scales[:, None], bias)

    torch.testing.assert_close(actual, expected)
    assert "weight" not in layer.state_dict()
    assert layer.cr8_scales.dtype == torch.uint8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the W8A8 kernel")
def test_convrot_linear_cuda_forward_quantizes_activations_to_int8():
    convrot = _convrot()
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(32, 64, generator=generator)
    bias = torch.randn(32, generator=generator, dtype=torch.bfloat16)
    qdata, scales, rot_size = convrot.quantize_weight(weight)
    layer = convrot.ConvRotLinear.from_quantized(
        qdata=qdata,
        scales=scales,
        bias=bias,
        rot_size=rot_size,
        original_dtype=torch.bfloat16,
    ).cuda()
    x = torch.randn(2, 3, 64, generator=generator, dtype=torch.bfloat16).cuda()

    actual = layer(x)
    rotated = convrot.rotate(x, rot_size).reshape(-1, 64)
    activation_codes, activation_scales = convrot.quantize_int8_rows(rotated)
    padded_rows = ((activation_codes.shape[0] + 31) // 32) * 32
    padded_codes = F.pad(activation_codes, (0, 0, 0, padded_rows - activation_codes.shape[0]))
    accum = torch._int_mm(padded_codes, layer.cr8_qdata.t())[: activation_codes.shape[0]]
    expected = (accum.float() * activation_scales[:, None] * layer.scales[None, :] + layer.bias.float()).to(torch.bfloat16)

    relative_l2 = torch.linalg.vector_norm(actual.reshape(-1, 32) - expected) / torch.linalg.vector_norm(expected)
    assert relative_l2.item() < 0.002
    assert layer.last_backend == "cuda_int8"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the fused activation quantizer")
def test_fused_activation_quantization_matches_reference_and_pads_rows():
    convrot = _convrot()
    values = torch.randn(19, 128, dtype=torch.bfloat16, device="cuda")

    actual_codes, actual_scales = convrot.quantize_int8_rows_fused(values)
    expected_codes, expected_scales = convrot.quantize_int8_rows(values)

    assert actual_codes.shape == (32, 128)
    assert actual_scales.shape == (32,)
    code_delta = (actual_codes[:19].to(torch.int16) - expected_codes.to(torch.int16)).abs()
    assert code_delta.max().item() <= 1
    assert (code_delta != 0).sum().item() <= 2
    torch.testing.assert_close(actual_scales[:19], expected_scales)


class _TinyCheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wide = nn.Linear(64, 32)
        self.narrow = nn.Linear(32, 7)
        self.conv = nn.Conv2d(2, 3, kernel_size=1)

    def forward(self, x):
        return self.narrow(torch.tanh(self.wide(x)))


def _write_tiny_checkpoint(path: Path, model: nn.Module) -> None:
    path.mkdir()
    save_file(model.state_dict(), path / "model.safetensors")
    (path / "config.json").write_text('{"model_type": "tiny"}', encoding="utf-8")


def _quantize_tiny(source: Path, output: Path, **options):
    return _convrot().quantize_checkpoint(
        model=_TinyCheckpointModel(), source_dir=source, output_dir=output, device=torch.device("cpu"), **options,
    )


def _bundle_contents(path: Path) -> dict[str, bytes]:
    return {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}


@pytest.mark.parametrize("output_path", ["source", ".", "source/bundle"])
def test_overwrite_rejects_overlapping_checkpoint_paths(tmp_path, output_path):
    family = tmp_path / "paths"
    family.mkdir()
    source = family / "source"
    _write_tiny_checkpoint(source, _TinyCheckpointModel())
    original = _bundle_contents(source)

    with pytest.raises(ValueError, match="overlap"):
        _quantize_tiny(source, family / output_path, overwrite=True)

    assert _bundle_contents(source) == original
    assert not (source / "bundle").exists()


@pytest.mark.parametrize("has_previous_bundle", [False, True])
def test_overwrite_refuses_unowned_files(tmp_path, has_previous_bundle):
    source, output = tmp_path / "source", tmp_path / "out"
    _write_tiny_checkpoint(source, _TinyCheckpointModel())
    if has_previous_bundle:
        _quantize_tiny(source, output)
    else:
        output.mkdir()
    (output / "notes.txt").write_text("keep this", encoding="utf-8")
    original = _bundle_contents(output)

    with pytest.raises(FileExistsError):
        _quantize_tiny(source, output, overwrite=True)

    assert _bundle_contents(output) == original


def test_failed_overwrite_leaves_previous_bundle_intact(tmp_path):
    source, output = tmp_path / "source", tmp_path / "out"
    _write_tiny_checkpoint(source, _TinyCheckpointModel())
    _quantize_tiny(source, output)
    original = _bundle_contents(output)

    def interrupt_conversion(message):
        if message.startswith("Quantized"):
            raise RuntimeError("interrupted conversion")

    with pytest.raises(RuntimeError, match="interrupted conversion"):
        _quantize_tiny(source, output, overwrite=True, progress=interrupt_conversion)

    assert _bundle_contents(output) == original


def test_overwrite_rolls_back_when_publishing_the_manifest_fails(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "out"
    _write_tiny_checkpoint(source, _TinyCheckpointModel())
    _quantize_tiny(source, output, max_shard_bytes=1_024)
    previous = _bundle_contents(output)
    replace = os.replace

    def locked_manifest(src, dst):
        if Path(dst) == output / "convrot_config.json" and Path(src).parent.name == "bundle":
            raise PermissionError("manifest is locked")
        return replace(src, dst)

    monkeypatch.setattr(os, "replace", locked_manifest)
    with pytest.raises(PermissionError, match="locked"):
        _quantize_tiny(source, output, overwrite=True)

    assert _bundle_contents(output) == previous


def test_overwrite_rechecks_destination_before_publishing(tmp_path):
    source, output = tmp_path / "source", tmp_path / "out"
    _write_tiny_checkpoint(source, _TinyCheckpointModel())
    output.mkdir()

    def use_output_directory(_message):
        (output / "config.json").write_text("another application", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _quantize_tiny(source, output, overwrite=True, progress=use_output_directory)

    assert _bundle_contents(output) == {"config.json": b"another application"}


def test_overwrite_refuses_a_destination_redirected_during_conversion(tmp_path):
    source, output, elsewhere = tmp_path / "source", tmp_path / "out", tmp_path / "elsewhere"
    _write_tiny_checkpoint(source, _TinyCheckpointModel())
    elsewhere.mkdir()

    def redirect_output(_message):
        if not output.exists():
            try:
                output.symlink_to(elsewhere, target_is_directory=True)
            except OSError as exc:
                pytest.skip(f"Directory symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="redirect"):
        _quantize_tiny(source, output, overwrite=True, progress=redirect_output)

    assert not list(elsewhere.iterdir())


def test_overwrite_publishes_only_the_completed_replacement(tmp_path):
    source, output = tmp_path / "source", tmp_path / "out"
    _write_tiny_checkpoint(source, _TinyCheckpointModel())
    _quantize_tiny(source, output, max_shard_bytes=1_024)
    previous = _bundle_contents(output)
    assert len(list(output.glob("*.safetensors"))) > 1

    def check_previous_remains_readable(_message):
        assert _bundle_contents(output) == previous

    _quantize_tiny(source, output, overwrite=True, progress=check_previous_remains_readable)

    assert {p.name for p in output.glob("*.safetensors")} == {"model-00001-of-00001.safetensors"}
    loaded = _convrot().load_quantized_checkpoint(
        model=_TinyCheckpointModel(), bundle_dir=output, device=torch.device("cpu"),
    )
    assert isinstance(loaded.wide, _convrot().ConvRotLinear)


def test_checkpoint_quantization_writes_reloadable_sharded_bundle(tmp_path):
    convrot = _convrot()
    torch.manual_seed(23)
    reference_model = _TinyCheckpointModel().eval()
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "quantized"
    _write_tiny_checkpoint(source_dir, reference_model)

    manifest = convrot.quantize_checkpoint(
        model=_TinyCheckpointModel(),
        source_dir=source_dir,
        output_dir=output_dir,
        device=torch.device("cpu"),
        max_shard_bytes=1_024,
        chunk_bytes=512,
    )

    index = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert set(manifest["quantized_modules"]) == {"wide"}
    assert "wide.weight" not in index["weight_map"]
    assert "wide.cr8_qdata" in index["weight_map"]
    assert "wide.cr8_scales" in index["weight_map"]
    assert "narrow.weight" in index["weight_map"]
    assert len(set(index["weight_map"].values())) > 1
    assert (output_dir / "config.json").is_file()
    assert manifest["quantized_parameter_count"] == reference_model.wide.weight.numel()
    assert manifest["total_parameter_count"] == sum(tensor.numel() for tensor in reference_model.state_dict().values())
    assert manifest["quantized_parameter_fraction"] == pytest.approx(
        manifest["quantized_parameter_count"] / manifest["total_parameter_count"]
    )

    quantized_model = convrot.load_quantized_checkpoint(
        model=_TinyCheckpointModel(),
        bundle_dir=output_dir,
        device=torch.device("cpu"),
    ).eval()
    inputs = torch.randn(4, 64)
    expected = reference_model(inputs)
    actual = quantized_model(inputs)
    relative_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)

    assert isinstance(quantized_model.wide, convrot.ConvRotLinear)
    assert isinstance(quantized_model.narrow, nn.Linear)
    assert relative_l2.item() < 0.02


def test_checkpoint_quantization_opens_each_source_shard_only_twice(tmp_path, monkeypatch):
    convrot = _convrot()
    source_dir = tmp_path / "source"
    _write_tiny_checkpoint(source_dir, _TinyCheckpointModel())
    real_safe_open = convrot.safe_open
    open_count = 0

    def counting_safe_open(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return real_safe_open(*args, **kwargs)

    monkeypatch.setattr(convrot, "safe_open", counting_safe_open)
    convrot.quantize_checkpoint(
        model=_TinyCheckpointModel(),
        source_dir=source_dir,
        output_dir=tmp_path / "out",
        device=torch.device("cpu"),
        max_shard_bytes=1_024,
        chunk_bytes=512,
    )

    assert open_count <= 2


def test_cli_layerdiff_profile_has_streaming_defaults(tmp_path):
    convrot = _convrot()

    args = convrot.parse_args(
        [
            "quantize",
            "--profile",
            "layerdiff3d",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert args.model == "layerdifforg/seethroughv0.0.2_layerdiff3d"
    assert args.subfolder == "unet"
    assert args.model_class.endswith(":UNetFrameConditionModel")
    assert args.max_shard_bytes == 2 * 1024**3
    assert args.chunk_bytes == 256 * 1024**2


def test_manifest_summary_omits_per_module_details(tmp_path):
    convrot = _convrot()
    manifest = {
        "quantized_modules": {"wide": {"rot_size": 64}},
        "skipped_linear_modules": {"narrow": "shape"},
        "quantized_parameter_count": 2_048,
        "total_parameter_count": 4_096,
        "quantized_parameter_fraction": 0.5,
        "original_tensor_bytes": 8_192,
        "quantized_tensor_bytes": 5_120,
    }

    summary = convrot.summarize_manifest(manifest, tmp_path / "out")

    assert summary["quantized_modules"] == 1
    assert summary["skipped_linear_modules"] == 1
    assert summary["tensor_byte_reduction"] == pytest.approx(0.375)
    assert "wide" not in json.dumps(summary)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1024", 1024), ("256MiB", 256 * 1024**2), ("2GiB", 2 * 1024**3), ("1.5GB", 1_500_000_000)],
)
def test_parse_size_accepts_binary_and_decimal_units(value, expected):
    convrot = _convrot()

    assert convrot.parse_size(value) == expected
