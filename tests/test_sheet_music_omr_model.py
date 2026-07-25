import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeIO:
    def __init__(self, name: str, value_type: str, shape: list):
        self.name = name
        self.type = value_type
        self.shape = shape


def _valid_model_payload() -> dict:
    i2w = {str(token_id): f"token-{token_id}" for token_id in range(215)}
    special_ids = {
        "<pad>": 0,
        "<bos>": 100,
        "<eos>": 183,
        "<s>": 44,
        "<t>": 29,
        "<b>": 132,
    }
    for token, token_id in special_ids.items():
        i2w[str(token_id)] = token
    i2w.update(
        {
            "10": "*clefF4",
            "11": "*clefG2",
            "12": "*-",
            "13": "4",
            "14": "c",
            "15": "r",
            "16": ".",
            "17": "=",
            "18": "*^",
            "19": "*v",
        }
    )
    return {
        "i2w": i2w,
        "w2i": {token: int(token_id) for token_id, token in i2w.items()},
        "out_categories": 215,
        "maxlen": 7512,
        "max_length": 20,
        "bos_token_id": None,
        "eos_token_id": None,
        "pad_token_id": None,
    }


def _valid_preprocessor_payload() -> dict:
    return {
        "color": "RGB",
        "do_normalize": False,
        "do_rescale": True,
        "do_resize": True,
        "image_size": [1024, 1024],
        "input_layout": "NCHW",
        "interpolation": "bilinear",
        "rescale_factor": 1 / 255,
    }


class _EncoderSession:
    def __init__(self):
        self.run_count = 0

    @staticmethod
    def get_inputs():
        return [_FakeIO("pixel_values", "tensor(float)", [1, 3, 1024, 1024])]

    @staticmethod
    def get_outputs():
        return [
            _FakeIO("raw_features", "tensor(float)", [1, 4096, 256]),
            _FakeIO("enhanced_features", "tensor(float)", [1, 4096, 256]),
        ]

    @staticmethod
    def get_providers():
        return ["CPUExecutionProvider"]

    def run(self, output_names, feeds):
        self.run_count += 1
        assert output_names == ["raw_features", "enhanced_features"]
        assert feeds["pixel_values"].shape == (1, 3, 1024, 1024)
        features = np.zeros((1, 4096, 256), dtype=np.float32)
        return [features, features + np.float32(1.0)]


class _DecoderSession:
    def __init__(self, predicted_ids):
        self.predicted_ids = list(predicted_ids)
        self.prefixes = []

    @staticmethod
    def get_inputs():
        return [
            _FakeIO("raw_features", "tensor(float)", [1, 4096, 256]),
            _FakeIO("enhanced_features", "tensor(float)", [1, 4096, 256]),
            _FakeIO("token_ids", "tensor(int64)", [1, "sequence_length"]),
        ]

    @staticmethod
    def get_outputs():
        return [_FakeIO("next_token_logits", "tensor(float)", [1, 215])]

    def run(self, output_names, feeds):
        assert output_names == ["next_token_logits"]
        self.prefixes.append(feeds["token_ids"].copy())
        ranked_ids = self.predicted_ids[len(self.prefixes) - 1]
        if isinstance(ranked_ids, int):
            ranked_ids = [ranked_ids]
        logits = np.full((1, 215), -np.inf, dtype=np.float32)
        for rank, token_id in enumerate(ranked_ids):
            logits[0, token_id] = np.float32(1000.0 - rank)
        return [logits]


def test_model_config_uses_custom_vocab_and_maxlen_not_hf_generation_fields():
    from module.sheet_music_omr.model import parse_model_config

    config = parse_model_config(_valid_model_payload())

    assert config.max_length == 7512
    assert config.pad_token_id == 0
    assert config.bos_token_id == 100
    assert config.eos_token_id == 183


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload["w2i"].pop("<bos>"), "w2i.*<bos>"),
        (lambda payload: payload.__setitem__("maxlen", 20), "maxlen"),
        (lambda payload: payload.__setitem__("maxlen", 7512.0), "maxlen"),
        (lambda payload: payload.__setitem__("out_categories", 214), "out_categories"),
        (
            lambda payload: payload.__setitem__("out_categories", 215.0),
            "out_categories",
        ),
        (lambda payload: payload["i2w"].pop("214"), "215"),
        (lambda payload: payload["w2i"].__setitem__("<t>", 44), "<t>"),
    ],
)
def test_model_config_rejects_bundle_drift(mutation, match: str):
    from module.sheet_music_omr.model import parse_model_config

    payload = _valid_model_payload()
    mutation(payload)

    with pytest.raises((KeyError, ValueError), match=match):
        parse_model_config(payload)


def test_runtime_validates_graph_contract_before_generation():
    from module.sheet_music_omr.model import MuSViTOnnxRuntime, parse_model_config
    from module.sheet_music_omr.preprocess import parse_preprocessor_config

    encoder = _EncoderSession()
    encoder.get_outputs = lambda: [
        _FakeIO("wrong_name", "tensor(float)", [1, 4096, 256]),
        _FakeIO("enhanced_features", "tensor(float)", [1, 4096, 256]),
    ]

    with pytest.raises(ValueError, match="encoder output contract mismatch"):
        MuSViTOnnxRuntime(
            encoder_session=encoder,
            decoder_session=_DecoderSession([183]),
            model_config=parse_model_config(_valid_model_payload()),
            preprocessor_config=parse_preprocessor_config(_valid_preprocessor_payload()),
        )

    assert encoder.run_count == 0


def test_constrained_greedy_generation_preserves_valid_argmax_sequence():
    from module.sheet_music_omr.model import MuSViTOnnxRuntime, parse_model_config
    from module.sheet_music_omr.preprocess import parse_preprocessor_config

    encoder = _EncoderSession()
    decoder = _DecoderSession([10, 29, 11, 132, 12, 29, 12, 132, 183])
    runtime = MuSViTOnnxRuntime(
        encoder_session=encoder,
        decoder_session=decoder,
        model_config=parse_model_config(_valid_model_payload()),
        preprocessor_config=parse_preprocessor_config(_valid_preprocessor_payload()),
    )

    result = runtime.generate_pixel_values(
        np.zeros((1, 3, 1024, 1024), dtype=np.float32),
        max_tokens=12,
    )

    assert encoder.run_count == 1
    assert [prefix.tolist() for prefix in decoder.prefixes] == [
        [[100]],
        [[100, 10]],
        [[100, 10, 29]],
        [[100, 10, 29, 11]],
        [[100, 10, 29, 11, 132]],
        [[100, 10, 29, 11, 132, 12]],
        [[100, 10, 29, 11, 132, 12, 29]],
        [[100, 10, 29, 11, 132, 12, 29, 12]],
        [[100, 10, 29, 11, 132, 12, 29, 12, 132]],
    ]
    assert result.token_ids == (100, 10, 29, 11, 132, 12, 29, 12, 132, 183)
    assert result.tokens == (
        "*clefF4",
        "<t>",
        "*clefG2",
        "<b>",
        "*-",
        "<t>",
        "*-",
        "<b>",
    )
    assert result.terminated_by_eos is True
    assert result.truncated is False


def test_constrained_greedy_accepts_complete_pinned_training_fixture():
    from module.sheet_music_omr.constraints import KernGreedyConstraint

    tokens = tuple(
        json.loads(
            (FIXTURES / "musvit_polish_scores_val0.json").read_text(
                encoding="utf-8"
            )
        )["tokens"]
    )
    vocabulary = tuple(dict.fromkeys((*tokens, "<eos>")))
    i2w = {token_id: token for token_id, token in enumerate(vocabulary)}
    w2i = {token: token_id for token_id, token in i2w.items()}
    constraint = KernGreedyConstraint(
        i2w,
        eos_token_id=w2i["<eos>"],
    )

    for token in tokens:
        token_id = w2i[token]
        assert constraint.is_allowed(token_id), (
            token,
            constraint.describe(),
        )
        constraint.accept(token_id)

    assert constraint.is_allowed(w2i["<eos>"])


def test_constrained_greedy_masks_early_row_break_without_extra_decoder_call():
    from module.sheet_music_omr.model import MuSViTOnnxRuntime, parse_model_config
    from module.sheet_music_omr.preprocess import parse_preprocessor_config

    decoder = _DecoderSession(
        [
            10,
            [132, 29],
            11,
            132,
            12,
            29,
            12,
            132,
            183,
        ]
    )
    runtime = MuSViTOnnxRuntime(
        encoder_session=_EncoderSession(),
        decoder_session=decoder,
        model_config=parse_model_config(_valid_model_payload()),
        preprocessor_config=parse_preprocessor_config(_valid_preprocessor_payload()),
    )

    result = runtime.generate_pixel_values(
        np.zeros((1, 3, 1024, 1024), dtype=np.float32),
        max_tokens=12,
    )

    assert result.token_ids[1:4] == (10, 29, 11)
    assert len(decoder.prefixes) == len(result.token_ids) - 1


def test_constrained_greedy_masks_separator_after_reciprocal_only_field():
    from module.sheet_music_omr.model import MuSViTOnnxRuntime, parse_model_config
    from module.sheet_music_omr.preprocess import parse_preprocessor_config

    decoder = _DecoderSession(
        [
            13,
            [29, 14],
            29,
            13,
            15,
            132,
            12,
            29,
            12,
            132,
            183,
        ]
    )
    runtime = MuSViTOnnxRuntime(
        encoder_session=_EncoderSession(),
        decoder_session=decoder,
        model_config=parse_model_config(_valid_model_payload()),
        preprocessor_config=parse_preprocessor_config(_valid_preprocessor_payload()),
    )

    result = runtime.generate_pixel_values(
        np.zeros((1, 3, 1024, 1024), dtype=np.float32),
        max_tokens=14,
    )

    assert result.tokens[:5] == ("4", "c", "<t>", "4", "r")


def test_constrained_greedy_masks_mixed_record_type_and_early_eos():
    from module.sheet_music_omr.model import MuSViTOnnxRuntime, parse_model_config
    from module.sheet_music_omr.preprocess import parse_preprocessor_config

    decoder = _DecoderSession(
        [
            [183, 13],
            14,
            29,
            [10, 13],
            15,
            132,
            12,
            29,
            12,
            132,
            183,
        ]
    )
    runtime = MuSViTOnnxRuntime(
        encoder_session=_EncoderSession(),
        decoder_session=decoder,
        model_config=parse_model_config(_valid_model_payload()),
        preprocessor_config=parse_preprocessor_config(_valid_preprocessor_payload()),
    )

    result = runtime.generate_pixel_values(
        np.zeros((1, 3, 1024, 1024), dtype=np.float32),
        max_tokens=14,
    )

    assert result.tokens[:5] == ("4", "c", "<t>", "4", "r")
    assert result.terminated_by_eos is True


def test_constrained_greedy_fails_when_no_finite_legal_token_exists():
    from module.sheet_music_omr.model import (
        KernConstraintError,
        MuSViTOnnxRuntime,
        parse_model_config,
    )
    from module.sheet_music_omr.preprocess import parse_preprocessor_config

    runtime = MuSViTOnnxRuntime(
        encoder_session=_EncoderSession(),
        decoder_session=_DecoderSession([[183]]),
        model_config=parse_model_config(_valid_model_payload()),
        preprocessor_config=parse_preprocessor_config(_valid_preprocessor_payload()),
    )

    with pytest.raises(KernConstraintError, match="no finite legal token"):
        runtime.generate_pixel_values(
            np.zeros((1, 3, 1024, 1024), dtype=np.float32),
            max_tokens=3,
        )


def test_generation_at_effective_limit_is_truncated():
    from module.sheet_music_omr.model import MuSViTOnnxRuntime, parse_model_config
    from module.sheet_music_omr.preprocess import parse_preprocessor_config

    runtime = MuSViTOnnxRuntime(
        encoder_session=_EncoderSession(),
        decoder_session=_DecoderSession([10, 29]),
        model_config=parse_model_config(_valid_model_payload()),
        preprocessor_config=parse_preprocessor_config(_valid_preprocessor_payload()),
    )

    result = runtime.generate_pixel_values(
        np.zeros((1, 3, 1024, 1024), dtype=np.float32),
        max_tokens=3,
    )

    assert result.token_ids == (100, 10, 29)
    assert result.terminated_by_eos is False
    assert result.truncated is True


def test_generation_rejects_unknown_predicted_token_id():
    from module.sheet_music_omr.model import MuSViTOnnxRuntime, parse_model_config
    from module.sheet_music_omr.preprocess import parse_preprocessor_config

    decoder = _DecoderSession([183])
    runtime = MuSViTOnnxRuntime(
        encoder_session=_EncoderSession(),
        decoder_session=decoder,
        model_config=parse_model_config(_valid_model_payload()),
        preprocessor_config=parse_preprocessor_config(_valid_preprocessor_payload()),
    )
    runtime.model_config.i2w.pop(183)

    with pytest.raises(KeyError, match="Unknown predicted token id 183"):
        runtime.generate_pixel_values(
            np.zeros((1, 3, 1024, 1024), dtype=np.float32),
            max_tokens=3,
        )


def test_recognizer_builds_pinned_two_graph_bundle(tmp_path: Path):
    from module.sheet_music_omr.model import (
        DEFAULT_MODEL_REPO_ID,
        DEFAULT_MODEL_REVISION,
        MuSViTOnnxRecognizer,
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_valid_model_payload()), encoding="utf-8")
    preprocessor_path = tmp_path / "preprocessor_config.json"
    preprocessor_path.write_text(json.dumps(_valid_preprocessor_payload()), encoding="utf-8")
    validation_path = tmp_path / "val_evaluation.json"
    validation_path.write_text("{}", encoding="utf-8")
    encoder_path = tmp_path / "encoder.onnx"
    encoder_path.write_bytes(b"fake")
    decoder_path = tmp_path / "decoder.onnx"
    decoder_path.write_bytes(b"fake")
    captured = {}

    def bundle_loader(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            artifact_paths={"encoder": encoder_path, "decoder": decoder_path},
            support_paths={
                "config": config_path,
                "preprocessor_config": preprocessor_path,
                "validation": validation_path,
            },
            sessions={"encoder": _EncoderSession(), "decoder": _DecoderSession([183])},
            providers=("CPUExecutionProvider",),
        )

    recognizer = MuSViTOnnxRecognizer(
        model_dir=tmp_path,
        bundle_loader=bundle_loader,
        verify_pinned_artifacts=False,
    )

    spec = captured["spec"]
    assert spec.repo_id == DEFAULT_MODEL_REPO_ID
    assert spec.revision == DEFAULT_MODEL_REVISION
    assert spec.artifacts == {"encoder": "encoder.onnx", "decoder": "decoder.onnx"}
    assert spec.support_files == {
        "config": "config.json",
        "preprocessor_config": "preprocessor_config.json",
        "validation": "val_evaluation.json",
    }
    assert recognizer.providers == ("CPUExecutionProvider",)
    assert set(recognizer.bundle_hashes) == {
        "encoder",
        "decoder",
        "config",
        "preprocessor_config",
    }
