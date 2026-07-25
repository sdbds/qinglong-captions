from pathlib import Path

import pytest

from utils import onnx_export


def test_parse_args_keeps_legacy_roformer_default(tmp_path):
    args = onnx_export.parse_args(["--model-dir", str(tmp_path), "--print-only"])

    assert args.target == "roformer"
    assert args.model_dir == tmp_path
    assert args.print_only is True


def test_obsolete_musvit_embedding_export_subcommand_is_rejected():
    with pytest.raises(SystemExit):
        onnx_export.parse_args(["musvit", "--print-only"])
