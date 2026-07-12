from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class ModelVariant(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

    @property
    def repo_id(self) -> str:
        return f"MuScriptor/muscriptor-{self.value}"


class DecodingMode(str, Enum):
    GREEDY = "greedy"
    SAMPLING = "sampling"
    BEAM = "beam"


class OutputFormat(str, Enum):
    MIDI = "midi"
    JSON = "json"
    JSONL = "jsonl"


class PreviewContent(str, Enum):
    MIDI = "midi"
    COMPARISON = "comparison"


class PreviewFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"


DEFAULT_MODEL = ModelVariant.MEDIUM
DEFAULT_DEVICE = "auto"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_CFG_COEF = 1.0
DEFAULT_BEAM_SIZE = 1
DEFAULT_BEAM_SEARCH_SIZE = 2
DEFAULT_OUTPUT_FORMATS = (OutputFormat.MIDI,)
DEFAULT_PREVIEW_FORMAT = PreviewFormat.WAV

_DEVICE_PATTERN = re.compile(r"^(?:auto|cpu|cuda(?::\d+)?)$")


def _enum_value(enum_type: type[Enum], value: object, field_name: str):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(str(item.value) for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {choices}") from exc


def _positive_int(value: int | None, field_name: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _normalize_instruments(values: Iterable[str] | None) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values or ():
        name = str(value).strip()
        if not name:
            continue
        if name not in normalized:
            normalized.append(name)
    return tuple(normalized)


@dataclass(frozen=True)
class PreviewRequest:
    content: PreviewContent
    format: PreviewFormat = DEFAULT_PREVIEW_FORMAT

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", _enum_value(PreviewContent, self.content, "preview content"))
        object.__setattr__(self, "format", _enum_value(PreviewFormat, self.format, "preview format"))

    def as_dict(self) -> dict[str, str]:
        return {"content": self.content.value, "format": self.format.value}


@dataclass(frozen=True)
class TranscriptionOptions:
    model: ModelVariant = DEFAULT_MODEL
    device: str = DEFAULT_DEVICE
    batch_size: int | None = None
    decode_mode: DecodingMode = DecodingMode.GREEDY
    temperature: float = DEFAULT_TEMPERATURE
    cfg_coef: float = DEFAULT_CFG_COEF
    strict_eos: bool = False
    beam_size: int = DEFAULT_BEAM_SIZE
    instruments: tuple[str, ...] = ()
    print_notes: bool = False

    def __post_init__(self) -> None:
        model = _enum_value(ModelVariant, self.model, "model")
        decode_mode = _enum_value(DecodingMode, self.decode_mode, "decode mode")
        device = str(self.device).strip().lower()
        if not _DEVICE_PATTERN.fullmatch(device):
            raise ValueError("device must be auto, cpu, cuda, or cuda:N")

        batch_size = _positive_int(self.batch_size, "batch size", allow_none=True)
        beam_size = _positive_int(self.beam_size, "beam size")
        temperature = float(self.temperature)
        cfg_coef = float(self.cfg_coef)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive number")
        if not math.isfinite(cfg_coef):
            raise ValueError("cfg coefficient must be finite")
        if decode_mode is DecodingMode.BEAM and beam_size < 2:
            raise ValueError("beam decoding requires beam size >= 2")
        if decode_mode is not DecodingMode.BEAM and beam_size != 1:
            raise ValueError("beam size must be 1 outside beam decoding")
        if decode_mode is not DecodingMode.SAMPLING and temperature != DEFAULT_TEMPERATURE:
            raise ValueError("temperature is only valid with sampling decoding")

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "decode_mode", decode_mode)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "beam_size", beam_size)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "cfg_coef", cfg_coef)
        object.__setattr__(self, "instruments", _normalize_instruments(self.instruments))

    @classmethod
    def from_single_cli(
        cls,
        *,
        model: ModelVariant | str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        batch_size: int | None = None,
        sampling: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
        cfg_coef: float = DEFAULT_CFG_COEF,
        strict_eos: bool = False,
        beam_size: int = DEFAULT_BEAM_SIZE,
        instruments: Iterable[str] | None = None,
        print_notes: bool = False,
    ) -> "TranscriptionOptions":
        if sampling and beam_size >= 2:
            raise ValueError("sampling and beam decoding cannot be enabled together")
        if beam_size >= 2:
            mode = DecodingMode.BEAM
        elif sampling:
            mode = DecodingMode.SAMPLING
        else:
            mode = DecodingMode.GREEDY
        return cls(
            model=model,
            device=device,
            batch_size=batch_size,
            decode_mode=mode,
            temperature=temperature,
            cfg_coef=cfg_coef,
            strict_eos=strict_eos,
            beam_size=beam_size,
            instruments=_normalize_instruments(instruments),
            print_notes=print_notes,
        )

    @classmethod
    def from_batch_cli(
        cls,
        *,
        model: ModelVariant | str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        batch_size: int | None = None,
        decode_mode: DecodingMode | str = DecodingMode.GREEDY,
        temperature: float = DEFAULT_TEMPERATURE,
        cfg_coef: float = DEFAULT_CFG_COEF,
        strict_eos: bool = False,
        beam_size: int | None = None,
        instruments: Iterable[str] | None = None,
        print_notes: bool = False,
    ) -> "TranscriptionOptions":
        mode = _enum_value(DecodingMode, decode_mode, "decode mode")
        resolved_beam_size = beam_size
        if resolved_beam_size is None:
            resolved_beam_size = DEFAULT_BEAM_SEARCH_SIZE if mode is DecodingMode.BEAM else DEFAULT_BEAM_SIZE
        return cls(
            model=model,
            device=device,
            batch_size=batch_size,
            decode_mode=mode,
            temperature=temperature,
            cfg_coef=cfg_coef,
            strict_eos=strict_eos,
            beam_size=resolved_beam_size,
            instruments=_normalize_instruments(instruments),
            print_notes=print_notes,
        )

    def upstream_kwargs(self) -> dict[str, object]:
        return {
            "use_sampling": self.decode_mode is DecodingMode.SAMPLING,
            "temperature": self.temperature,
            "cfg_coef": self.cfg_coef,
            "instruments": list(self.instruments) or None,
            "batch_size": self.batch_size,
            "no_eos_is_ok": not self.strict_eos,
            "beam_size": self.beam_size,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model.value,
            "device": self.device,
            "batch_size": self.batch_size,
            "decode_mode": self.decode_mode.value,
            "temperature": self.temperature,
            "cfg_coef": self.cfg_coef,
            "strict_eos": self.strict_eos,
            "beam_size": self.beam_size,
            "instruments": list(self.instruments),
            "print_notes": self.print_notes,
        }


@dataclass(frozen=True)
class BatchOptions:
    transcription: TranscriptionOptions = field(default_factory=TranscriptionOptions)
    output_formats: tuple[OutputFormat, ...] = DEFAULT_OUTPUT_FORMATS
    preview: PreviewRequest | None = None
    recursive: bool = True
    skip_completed: bool = True
    overwrite: bool = False
    fail_fast: bool = False

    def __post_init__(self) -> None:
        formats: list[OutputFormat] = []
        for raw_format in self.output_formats:
            output_format = _enum_value(OutputFormat, raw_format, "output format")
            if output_format not in formats:
                formats.append(output_format)
        if not formats:
            raise ValueError("at least one symbolic output is required")
        if self.preview is not None and not isinstance(self.preview, PreviewRequest):
            raise TypeError("preview must be a PreviewRequest or None")
        object.__setattr__(self, "output_formats", tuple(formats))

    def as_dict(self) -> dict[str, object]:
        return {
            "transcription": self.transcription.as_dict(),
            "output_formats": [item.value for item in self.output_formats],
            "preview": self.preview.as_dict() if self.preview else None,
            "recursive": self.recursive,
            "skip_completed": self.skip_completed,
            "overwrite": self.overwrite,
            "fail_fast": self.fail_fast,
        }
