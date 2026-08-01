import subprocess
import sys
from pathlib import Path

import module.auto_rig as auto_rig

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_PUBLIC_API = {
    "AUTO_RIG_INPUT_CONTRACT_VERSION",
    "ArtifactContractError",
    "AutoRigCanvasContract",
    "AutoRigContractError",
    "AutoRigInputContract",
    "AutoRigPartContract",
    "AutoRigTagContractError",
    "CanonicalPartTag",
    "CanonicalLabelMap",
    "DRAW_ORDER_POLICY_VERSION",
    "DrawOrderPolicyError",
    "ERROR_RECORD_PATH",
    "EXPORT_MANIFEST_PATH",
    "FileDigest",
    "FormatValidation",
    "LoadedPartAlpha",
    "MASK_COMPONENT_ID_SCHEMA",
    "MASK_COMPONENT_PLAN_VERSION",
    "MaskCleanupDescriptor",
    "MaskComponentPlan",
    "MaskComponentPlanError",
    "MaskComponentRecord",
    "NormalizedMaskPart",
    "OrdinaryDrawOrderPlan",
    "PartDrawOrderRecord",
    "QCL_CODEC_VERSION",
    "QclContractError",
    "StageFailureRecord",
    "StageGraphContractError",
    "StageGraphResult",
    "StageGraphValidator",
    "StageManifest",
    "StageManifestError",
    "StageNode",
    "StageValidationIssue",
    "SUPPORTED_CANVAS_EDGES",
    "TerminalFinalizationError",
    "TerminalFinalizationResult",
    "ValidatedPartSource",
    "atomic_write_bytes",
    "atomic_write_json",
    "build_stage_fingerprint",
    "build_stage_manifest",
    "build_base_mask_component_plan",
    "build_ordinary_draw_order",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "describe_file",
    "decode_qcl",
    "encode_qcl",
    "finalize_failure",
    "finalize_success",
    "invalidate_terminal",
    "is_item_completed",
    "load_auto_rig_input_contract",
    "load_validated_part_alphas",
    "manifest_relative_path",
    "normalize_relative_path",
    "read_stage_manifest",
    "sha256_file",
    "validate_draw_order_registry",
    "validate_v3_final_tag_set",
    "write_stage_manifest",
}


def test_auto_rig_exports_only_the_supported_foundation_api() -> None:
    assert set(auto_rig.__all__) == EXPECTED_PUBLIC_API
    assert all(hasattr(auto_rig, name) for name in EXPECTED_PUBLIC_API)


def test_auto_rig_import_has_no_heavy_runtime_dependencies() -> None:
    script = """
import sys
import module.auto_rig
heavy = sorted(
    name
    for name in ('torch', 'cv2', 'psd_tools', 'numpy', 'scipy', 'skimage')
    if name in sys.modules
)
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
