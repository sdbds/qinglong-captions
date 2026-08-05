from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from module.auto_rig import cli


def test_auto_rig_cli_has_one_positional_item_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[Path, dict[str, object]]] = []
    monkeypatch.setenv("CUBISM_SDK_ROOT", "C:/sdk/CubismSdkForNative-5-r.5")
    monkeypatch.setenv("LIVE2D_RENDERER_PATH", "C:/ignored/legacy-validator.exe")
    monkeypatch.setattr(
        cli,
        "run_auto_rig_item",
        lambda item, **kwargs: (
            calls.append((Path(item), kwargs))
            or SimpleNamespace(
                item_root=tmp_path,
                reused_stages=("A", "B"),
                terminal=SimpleNamespace(payload={"status": "completed"}),
            )
        ),
    )

    assert cli.main([str(tmp_path)]) == 0

    assert calls == [
        (
            tmp_path,
            {
                "profile_id": "dual_runtime_core_v1",
                "validation_tier": "release",
                "sdk_root": "C:/sdk/CubismSdkForNative-5-r.5",
                "spine_runtime_path": None,
                "pose_mode": "auto",
                "pose_model_cache_dir": None,
                "sdpose_bundle_path": None,
                "detrpose_weights_path": None,
                "pose_device": None,
                "prefer_pose_fa2": True,
                "finalize": True,
            },
        )
    ]
    assert '"status": "completed"' in capsys.readouterr().out


def test_auto_rig_cli_rejects_a_second_item_argument(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args([str(tmp_path), str(tmp_path)])


def test_auto_rig_cli_stops_at_stage_d_for_the_spine_dev_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("AUTO_RIG_PROFILE", "spine_4_2_dev")
    monkeypatch.setattr(
        cli,
        "run_auto_rig_item",
        lambda _item, **kwargs: (
            calls.append(kwargs)
            or SimpleNamespace(
                item_root=tmp_path,
                reused_stages=("A", "B", "C", "D"),
                terminal=None,
            )
        ),
    )

    assert cli.main([str(tmp_path)]) == 0
    assert calls[0]["finalize"] is False
