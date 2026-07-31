import subprocess
import sys
from pathlib import Path

import module.auto_rig as auto_rig

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_PUBLIC_API = {
    "ArtifactContractError",
    "ERROR_RECORD_PATH",
    "EXPORT_MANIFEST_PATH",
    "FileDigest",
    "FormatValidation",
    "StageFailureRecord",
    "StageGraphContractError",
    "StageGraphResult",
    "StageGraphValidator",
    "StageManifest",
    "StageManifestError",
    "StageNode",
    "StageValidationIssue",
    "TerminalFinalizationError",
    "TerminalFinalizationResult",
    "atomic_write_bytes",
    "atomic_write_json",
    "build_stage_fingerprint",
    "build_stage_manifest",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "describe_file",
    "finalize_failure",
    "finalize_success",
    "invalidate_terminal",
    "is_item_completed",
    "manifest_relative_path",
    "normalize_relative_path",
    "read_stage_manifest",
    "sha256_file",
    "write_stage_manifest",
}


def test_auto_rig_exports_only_the_supported_foundation_api() -> None:
    assert set(auto_rig.__all__) == EXPECTED_PUBLIC_API
    assert all(hasattr(auto_rig, name) for name in EXPECTED_PUBLIC_API)


def test_auto_rig_import_has_no_heavy_runtime_dependencies() -> None:
    script = """
import sys
import module.auto_rig
heavy = sorted(name for name in ('torch', 'cv2', 'psd_tools', 'numpy') if name in sys.modules)
print(','.join(heavy))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == ""
