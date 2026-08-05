from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from module.auto_rig.export.live2d.e0_attestation import write_live2d_e0_attestation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the reviewed Live2D E0 frame attestation")
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-release", default="5-r.5")
    parser.add_argument("--platform-id")
    parser.add_argument("--backend-id")
    parser.add_argument(
        "--acknowledge-license-policy",
        action="store_true",
        help="Acknowledge that SDK/Core binaries are not packaged and licensing remains an external gate",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    write_live2d_e0_attestation(
        arguments.output,
        arguments.core,
        arguments.renderer,
        sdk_release=arguments.sdk_release,
        license_policy_acknowledged=arguments.acknowledge_license_policy,
        platform_id=arguments.platform_id,
        backend_id=arguments.backend_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
