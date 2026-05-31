import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_captioner_parser_accepts_cloud_max_concurrency():
    from module.captioner import setup_parser

    parser = setup_parser()

    default_args = parser.parse_args(["dataset"])
    assert default_args.cloud_max_concurrency == 1

    args = parser.parse_args(["dataset", "--cloud_max_concurrency=3"])
    assert args.cloud_max_concurrency == 3


def test_captioner_powershell_passes_cloud_max_concurrency_only_when_gt_one():
    script = (ROOT / "4.captioner.ps1").read_text(encoding="utf-8")

    assert "$cloud_max_concurrency = 1" in script
    assert "if ($cloud_max_concurrency -gt 1)" in script
    assert '--cloud_max_concurrency=$cloud_max_concurrency' in script
