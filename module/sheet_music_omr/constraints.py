"""Incremental syntax constraints for MuSViT BeKern greedy decoding."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .decode import KernStructureError, next_kern_spine_count

_PITCH_TOKEN = re.compile(
    r"^(?:A+|B+|C+|D+|E+|F+|G+|a+|b+|c+|d+|e+|f+|g+)$"
)
_CONTROL_TOKENS = frozenset({"<pad>", "<bos>", "<eos>", "<s>", "<t>", "<b>"})


class KernConstraintError(ValueError):
    """Raised when decoder logits contain no finite legal continuation."""


def _record_kind(token: str) -> str:
    if token.startswith("**"):
        return "exclusive"
    if token.startswith("*"):
        return "interpretation"
    if token.startswith("="):
        return "barline"
    return "data"


def _is_pitch_token(token: str) -> bool:
    return _PITCH_TOKEN.fullmatch(token) is not None


class KernGreedyConstraint:
    """Track the smallest prefix state needed to reject impossible separators."""

    def __init__(
        self,
        i2w: Mapping[int, str],
        *,
        eos_token_id: int,
    ) -> None:
        self.i2w = dict(i2w)
        self.eos_token_id = int(eos_token_id)
        self.active_spines = 2
        self.line_number = 1
        self.terminated = False
        self.row_kind: str | None = None
        self.completed_fields: list[tuple[str, ...]] = []
        self.field_tokens: list[str] = []

    def describe(self) -> str:
        return (
            f"line={self.line_number}, active_spines={self.active_spines}, "
            f"row_kind={self.row_kind!r}, completed_fields="
            f"{len(self.completed_fields)}, field_tokens={self.field_tokens!r}, "
            f"terminated={self.terminated}"
        )

    @staticmethod
    def _components(tokens: tuple[str, ...] | list[str]) -> list[list[str]]:
        components: list[list[str]] = [[]]
        for token in tokens:
            if token == "<s>":
                components.append([])
            else:
                components[-1].append(token)
        return components

    def _data_field_complete(self) -> bool:
        if self.field_tokens == ["."]:
            return True
        if not self.field_tokens or self.field_tokens[0] == ".":
            return False
        components = self._components(self.field_tokens)
        return all(
            component
            and (
                "r" in component
                or any(_is_pitch_token(token) for token in component)
            )
            for component in components
        )

    def _field_complete(self) -> bool:
        if self.row_kind in {"interpretation", "barline"}:
            return len(self.field_tokens) == 1
        if self.row_kind == "data":
            return self._data_field_complete()
        return False

    def _can_open_chord_component(self) -> bool:
        if self.row_kind != "data" or not self.field_tokens:
            return False
        if self.field_tokens[0] == ".":
            return False
        component = self._components(self.field_tokens)[-1]
        return (
            bool(component)
            and "r" not in component
            and any(_is_pitch_token(token) for token in component)
        )

    def _row_fields(self) -> list[str]:
        fields = [self._decode_field(tokens) for tokens in self.completed_fields]
        fields.append(self._decode_field(tuple(self.field_tokens)))
        return fields

    @staticmethod
    def _decode_field(tokens: tuple[str, ...]) -> str:
        return "".join(" " if token == "<s>" else token for token in tokens)

    def _next_spine_count(self) -> int | None:
        if self.row_kind != "interpretation":
            return self.active_spines
        try:
            return next_kern_spine_count(
                self._row_fields(),
                line_number=self.line_number,
            )
        except KernStructureError:
            return None

    def _can_close_row(self) -> bool:
        if not self._field_complete():
            return False
        if len(self.completed_fields) + 1 != self.active_spines:
            return False
        return self._next_spine_count() is not None

    def _can_accept_data_token(self, token: str) -> bool:
        if token == ".":
            return not self.field_tokens or self.field_tokens[0] != "."
        if self.field_tokens and self.field_tokens[0] == ".":
            return False
        component = self._components(self.field_tokens)[-1]
        if token == "r" and any(_is_pitch_token(item) for item in component):
            return False
        if _is_pitch_token(token) and "r" in component:
            return False
        return True

    def is_allowed(self, token_id: int) -> bool:
        token = self.i2w.get(int(token_id))
        if token is None:
            return False
        if self.terminated:
            return int(token_id) == self.eos_token_id
        if token in {"<pad>", "<bos>", "<eos>"}:
            return False
        if token == "<s>":
            return self._can_open_chord_component()
        if token == "<t>":
            return (
                self._field_complete()
                and len(self.completed_fields) + 1 < self.active_spines
            )
        if token == "<b>":
            return self._can_close_row()
        if token in _CONTROL_TOKENS:
            return False

        kind = _record_kind(token)
        if kind == "exclusive":
            return False
        if self.row_kind is not None and kind != self.row_kind:
            return False
        if self.row_kind in {"interpretation", "barline"} and self.field_tokens:
            return False
        if kind == "data":
            return self._can_accept_data_token(token)
        return True

    def allowed_token_ids(self) -> tuple[int, ...]:
        return tuple(
            token_id
            for token_id in sorted(self.i2w)
            if self.is_allowed(token_id)
        )

    def accept(self, token_id: int) -> None:
        token_id = int(token_id)
        if not self.is_allowed(token_id):
            token = self.i2w.get(token_id, f"<unknown:{token_id}>")
            raise KernConstraintError(
                f"illegal Kern token {token!r}: {self.describe()}"
            )
        token = self.i2w[token_id]
        if token == "<eos>":
            return
        if token == "<t>":
            self.completed_fields.append(tuple(self.field_tokens))
            self.field_tokens = []
            return
        if token == "<b>":
            next_count = self._next_spine_count()
            if next_count is None:
                raise KernConstraintError(
                    f"invalid Kern spine row: {self.describe()}"
                )
            self.active_spines = next_count
            self.terminated = next_count == 0
            self.line_number += 1
            self.row_kind = None
            self.completed_fields = []
            self.field_tokens = []
            return
        if token == "<s>":
            self.field_tokens.append(token)
            return

        if self.row_kind is None:
            self.row_kind = _record_kind(token)
        self.field_tokens.append(token)
