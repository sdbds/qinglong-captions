from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RewardCatalogError(RuntimeError):
    """Raised when the project reward-model environment cannot expose its catalog."""


@dataclass(frozen=True, slots=True)
class RewardCheckpoint:
    identifier: str
    is_default: bool
    tracks_updates: bool


@dataclass(frozen=True, slots=True)
class RewardCatalog:
    scorers: tuple[str, ...] = ()
    checkpoints: Mapping[str, tuple[RewardCheckpoint, ...]] = field(default_factory=dict)

    def checkpoints_for(self, scorer: str) -> tuple[RewardCheckpoint, ...]:
        return self.checkpoints.get(scorer, ())


def resolve_project_venv_python(project_root: str | Path = PROJECT_ROOT) -> Path:
    root = Path(project_root).resolve()
    relative = Path(".venv/Scripts/python.exe") if sys.platform == "win32" else Path(".venv/bin/python")
    python_path = root / relative
    if not python_path.is_file():
        raise RewardCatalogError(f"Project .venv Python executable not found: {python_path}")
    return python_path


def _parse_checkpoint(raw: Any, *, scorer: str) -> RewardCheckpoint:
    if not isinstance(raw, dict):
        raise RewardCatalogError(f"Invalid checkpoint metadata for scorer {scorer!r}")

    identifier = raw.get("identifier")
    is_default = raw.get("is_default")
    tracks_updates = raw.get("tracks_updates")
    if not isinstance(identifier, str) or not identifier.strip():
        raise RewardCatalogError(f"Invalid checkpoint identifier for scorer {scorer!r}")
    if type(is_default) is not bool or type(tracks_updates) is not bool:
        raise RewardCatalogError(f"Invalid checkpoint flags for scorer {scorer!r}")
    return RewardCheckpoint(identifier.strip(), is_default, tracks_updates)


def _parse_catalog(payload: str) -> RewardCatalog:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RewardCatalogError("Qinglong Score returned invalid catalog JSON") from error

    if not isinstance(raw, dict):
        raise RewardCatalogError("Qinglong Score catalog must be a JSON object")
    raw_scorers = raw.get("scorers")
    raw_checkpoints = raw.get("checkpoints")
    if not isinstance(raw_scorers, list) or not isinstance(raw_checkpoints, dict):
        raise RewardCatalogError("Qinglong Score catalog is missing required fields")

    scorers: list[str] = []
    seen: set[str] = set()
    for raw_scorer in raw_scorers:
        if not isinstance(raw_scorer, str) or not raw_scorer.strip():
            raise RewardCatalogError("Qinglong Score returned an invalid scorer name")
        scorer = raw_scorer.strip()
        if scorer in seen:
            raise RewardCatalogError(f"Qinglong Score returned duplicate scorer {scorer!r}")
        seen.add(scorer)
        scorers.append(scorer)

    checkpoints: dict[str, tuple[RewardCheckpoint, ...]] = {}
    for scorer in scorers:
        rows = raw_checkpoints.get(scorer, [])
        if not isinstance(rows, list):
            raise RewardCatalogError(f"Invalid checkpoint list for scorer {scorer!r}")
        checkpoints[scorer] = tuple(_parse_checkpoint(row, scorer=scorer) for row in rows)
    return RewardCatalog(tuple(scorers), checkpoints)


def load_reward_catalog(
    *,
    project_root: str | Path = PROJECT_ROOT,
    timeout: float = 30.0,
) -> RewardCatalog:
    root = Path(project_root).resolve()
    python_path = resolve_project_venv_python(root)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "VIRTUAL_ENV": str(root / ".venv"),
        }
    )
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    try:
        completed = subprocess.run(
            [str(python_path), "-m", "gui.utils.reward_catalog"],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired as error:
        raise RewardCatalogError("Qinglong Score catalog discovery timed out") from error
    except OSError as error:
        raise RewardCatalogError(f"Unable to start project .venv Python: {error}") from error

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        if detail:
            detail = detail.splitlines()[-1]
            raise RewardCatalogError(f"Qinglong Score catalog discovery failed: {detail}")
        raise RewardCatalogError("Qinglong Score catalog discovery failed")
    return _parse_catalog(completed.stdout)


def _build_catalog_payload() -> dict[str, Any]:
    import qinglong_score

    scorers = list(qinglong_score.list_scorers())
    checkpoints = {}
    for scorer in scorers:
        checkpoints[scorer] = [
            {
                "identifier": row.identifier,
                "is_default": row.is_default,
                "tracks_updates": row.tracks_updates,
            }
            for row in qinglong_score.list_checkpoints(scorer)
        ]
    return {"scorers": scorers, "checkpoints": checkpoints}


def _main() -> int:
    try:
        payload = _build_catalog_payload()
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
