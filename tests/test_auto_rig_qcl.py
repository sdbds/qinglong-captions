from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from module.auto_rig.qcl import (
    CanonicalLabelMap,
    QclContractError,
    decode_qcl,
    encode_qcl,
    materialize_qcl,
)


def test_qcl1_has_exact_little_endian_bytes_and_round_trips() -> None:
    labels = (0, 1, 1, 0, 2, 2)
    expected = b"QCL1" + struct.pack("<II6I", 3, 2, *labels)

    payload = encode_qcl(labels, width=3, height=2)

    assert payload == expected
    assert decode_qcl(payload) == CanonicalLabelMap(
        width=3,
        height=2,
        labels=labels,
    )


@pytest.mark.parametrize(
    "payload",
    (
        b"BAD!" + struct.pack("<II4I", 2, 2, 0, 1, 1, 0),
        b"QCL1" + struct.pack(">II4I", 2, 2, 0, 1, 1, 0),
        b"QCL1" + struct.pack("<II3I", 2, 2, 0, 1, 1),
        b"QCL1" + struct.pack("<II4I", 2, 2, 0, 1, 1, 0) + b"\x00",
        b"QCL1" + struct.pack("<II4I", 2, 2, 0, 2, 2, 0),
    ),
)
def test_qcl1_rejects_corrupt_or_noncanonical_payloads(payload: bytes) -> None:
    with pytest.raises(QclContractError):
        decode_qcl(payload)


@pytest.mark.parametrize(
    ("labels", "width", "height"),
    (
        ((0, 1, 1), 2, 2),
        ((0, 2, 2, 0), 2, 2),
        ((0, 0, 0, 0), 2, 2),
        ((0, True, 1, 0), 2, 2),
        ((0, -1, 1, 0), 2, 2),
        ((0, 1, 1, 0), 0, 2),
    ),
)
def test_qcl1_encoder_rejects_invalid_shapes_and_labels(
    labels: tuple[object, ...],
    width: int,
    height: int,
) -> None:
    with pytest.raises(QclContractError):
        encode_qcl(labels, width=width, height=height)
