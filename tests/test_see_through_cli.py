import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from module.see_through.cli import build_parser, build_run_config
from module.see_through.see_through_profile import SEE_THROUGH_REPO_MAP


def test_see_through_parser_defaults_from_config():
    parser = build_parser()

    args = parser.parse_args(["--input_dir=foo"])

    assert args.repo_id_layerdiff == "layerdifforg/seethroughv0.0.2_layerdiff3d"
    assert args.repo_id_depth == "24yearsold/seethroughv0.0.1_marigold"
    assert args.resolution == 1024
    assert args.resolution_depth == 720
    assert args.inference_steps_depth == -1
    assert args.seed == 42
    assert args.quant_mode == "none"
    assert args.group_offload is False
    assert args.offload_policy == "delete"
    assert args.skip_completed is True
    assert args.save_to_psd is True
    assert args.output_dir == "workspace/see_through_output"


def test_build_run_config_switches_known_repo_family_for_nf4():
    parser = build_parser()

    args = parser.parse_args(["--input_dir=foo", "--quant_mode=nf4"])
    config = build_run_config(args)

    assert config.quant_mode == "nf4"
    assert config.repo_id_layerdiff == SEE_THROUGH_REPO_MAP["nf4"]["layerdiff"]
    assert config.repo_id_depth == SEE_THROUGH_REPO_MAP["nf4"]["depth"]


def test_build_run_config_preserves_seed_and_depth_steps():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--input_dir=foo",
            "--seed=123",
            "--inference_steps_depth=7",
        ]
    )
    config = build_run_config(args)

    assert config.seed == 123
    assert config.inference_steps_depth == 7


def test_see_through_script_help_runs_from_script_path():
    result = subprocess.run(
        [sys.executable, str(ROOT / "module" / "see_through" / "cli.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "repo_id_layerdiff" in result.stdout
    assert "output_dir" in result.stdout
    assert "skip_completed" in result.stdout
    assert "quant_mode" in result.stdout
    assert "resolution_depth" in result.stdout
    assert "inference_steps_depth" in result.stdout
    assert "seed" in result.stdout
    assert "group_offload" in result.stdout


def test_see_through_module_directory_does_not_shadow_stdlib_profile():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, r'{ROOT / 'module' / 'see_through'}'); "
                "import cProfile; "
                "print(cProfile.__file__)"
            ),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )

    assert result.returncode == 0, result.stderr
