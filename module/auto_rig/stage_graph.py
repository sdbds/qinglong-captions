from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .artifacts import ArtifactContractError, describe_file, sha256_file
from .manifests import (
    VALID_STAGE_NAMES,
    StageManifest,
    StageManifestError,
    manifest_relative_path,
    read_stage_manifest,
)

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class StageGraphContractError(ValueError):
    """Raised for a programmer or startup configuration error in the stage DAG."""


@dataclass(frozen=True)
class StageNode:
    stage_name: str
    upstream_stages: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.stage_name not in VALID_STAGE_NAMES:
            raise StageGraphContractError(f"unknown stage node: {self.stage_name!r}")
        upstream = tuple(self.upstream_stages)
        if len(upstream) != len(set(upstream)):
            raise StageGraphContractError(f"stage {self.stage_name} contains duplicate dependencies")
        for dependency in upstream:
            if dependency not in VALID_STAGE_NAMES:
                raise StageGraphContractError(f"stage {self.stage_name} references unknown stage {dependency!r}")
            if dependency == self.stage_name:
                raise StageGraphContractError(f"stage {self.stage_name} cannot depend on itself")
        object.__setattr__(self, "upstream_stages", upstream)


DEFAULT_RELEASE_STAGE_NODES = (
    StageNode("A", ()),
    StageNode("B", ("A",)),
    StageNode("C", ("A", "B")),
    StageNode("D", ("C",)),
    StageNode("E", ("C",)),
    StageNode("G", ("C", "D", "E")),
)


@dataclass(frozen=True)
class StageValidationIssue:
    code: str
    stage_name: str
    path: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class StageGraphResult:
    target_stage: str
    reusable: bool
    issues: tuple[StageValidationIssue, ...]
    manifests: tuple[tuple[str, StageManifest], ...]

    def manifest(self, stage_name: str) -> StageManifest | None:
        return dict(self.manifests).get(stage_name)


class StageGraphValidator:
    def __init__(
        self,
        item_root: str | Path,
        *,
        nodes: Iterable[StageNode] = DEFAULT_RELEASE_STAGE_NODES,
    ) -> None:
        self.item_root = Path(item_root)
        normalized_nodes = tuple(nodes)
        if not normalized_nodes:
            raise StageGraphContractError("stage graph must contain at least one node")
        names = [node.stage_name for node in normalized_nodes]
        if len(names) != len(set(names)):
            raise StageGraphContractError("stage graph contains duplicate stage nodes")
        self._nodes = {node.stage_name: node for node in normalized_nodes}
        for node in normalized_nodes:
            missing = set(node.upstream_stages) - set(self._nodes)
            if missing:
                raise StageGraphContractError(f"stage {node.stage_name} references missing graph nodes: {sorted(missing)}")
        self._topological_order = self._build_topological_order(normalized_nodes)
        self._order_index = {stage: index for index, stage in enumerate(self._topological_order)}

    def _build_topological_order(self, nodes: tuple[StageNode, ...]) -> tuple[str, ...]:
        state: dict[str, int] = {}
        ordered: list[str] = []

        def visit(stage: str) -> None:
            marker = state.get(stage, 0)
            if marker == 1:
                raise StageGraphContractError(f"stage graph contains a dependency cycle at {stage}")
            if marker == 2:
                return
            state[stage] = 1
            for dependency in self._nodes[stage].upstream_stages:
                visit(dependency)
            state[stage] = 2
            ordered.append(stage)

        for node in nodes:
            visit(node.stage_name)
        return tuple(ordered)

    def _required_stages(self, target_stage: str) -> frozenset[str]:
        if target_stage not in self._nodes:
            raise StageGraphContractError(f"target stage is not present in the graph: {target_stage!r}")
        required: set[str] = set()

        def include(stage: str) -> None:
            if stage in required:
                return
            required.add(stage)
            for dependency in self._nodes[stage].upstream_stages:
                include(dependency)

        include(target_stage)
        return frozenset(required)

    def validate(
        self,
        *,
        target_stage: str = "G",
        expected_fingerprints: Mapping[str, str] | None = None,
    ) -> StageGraphResult:
        required = self._required_stages(target_stage)
        expected = self._normalize_expected_fingerprints(expected_fingerprints, required)
        manifests: dict[str, StageManifest] = {}
        issues: list[StageValidationIssue] = []

        for stage in self._topological_order:
            if stage not in required:
                continue
            marker_relative = manifest_relative_path(stage)
            marker = self.item_root / Path(*marker_relative.split("/"))
            if not marker.is_file():
                issues.append(
                    StageValidationIssue(
                        code="manifest_missing",
                        stage_name=stage,
                        path=marker_relative,
                        detail="stage commit marker is missing",
                    )
                )
                continue
            try:
                manifest = read_stage_manifest(self.item_root, stage)
            except StageManifestError as exc:
                issues.append(
                    StageValidationIssue(
                        code="manifest_invalid",
                        stage_name=stage,
                        path=marker_relative,
                        detail=str(exc),
                    )
                )
                continue
            manifests[stage] = manifest

            expected_fingerprint = expected.get(stage)
            if expected_fingerprint is None:
                issues.append(
                    StageValidationIssue(
                        code="expected_fingerprint_missing",
                        stage_name=stage,
                        path=marker_relative,
                        detail="current code/config did not provide a stage fingerprint",
                    )
                )
            elif manifest.stage_fingerprint != expected_fingerprint:
                issues.append(
                    StageValidationIssue(
                        code="stage_fingerprint_mismatch",
                        stage_name=stage,
                        path=marker_relative,
                        detail="stored stage fingerprint differs from current code/config",
                    )
                )

            if manifest.status != "completed":
                issues.append(
                    StageValidationIssue(
                        code="stage_status_not_completed",
                        stage_name=stage,
                        path=marker_relative,
                        detail=f"stored status is {manifest.status!r}",
                    )
                )

            expected_upstreams = frozenset(self._nodes[stage].upstream_stages)
            actual_upstreams = frozenset(dict(manifest.upstream_manifests))
            if actual_upstreams != expected_upstreams:
                issues.append(
                    StageValidationIssue(
                        code="upstream_set_mismatch",
                        stage_name=stage,
                        path=marker_relative,
                        detail=(f"expected {sorted(expected_upstreams)}, got {sorted(actual_upstreams)}"),
                    )
                )

            self._validate_outputs(manifest, issues)

        self._validate_upstream_marker_digests(required, manifests, issues)
        self._validate_output_ownership(manifests, issues)

        terminal = manifests.get("G") if target_stage == "G" else None
        if terminal is not None and terminal.status == "completed":
            terminal_dependencies = frozenset(dict(terminal.upstream_manifests))
            required_terminal_dependencies = frozenset({"C", "D", "E"})
            if terminal_dependencies != required_terminal_dependencies:
                issues.append(
                    StageValidationIssue(
                        code="terminal_dependencies_missing",
                        stage_name="G",
                        path=manifest_relative_path("G"),
                        detail=("completed G must bind exactly the current C, D, and E manifests"),
                    )
                )

        sorted_issues = tuple(sorted(issues, key=self._issue_sort_key))
        loaded = tuple((stage, manifests[stage]) for stage in self._topological_order if stage in required and stage in manifests)
        return StageGraphResult(
            target_stage=target_stage,
            reusable=not sorted_issues,
            issues=sorted_issues,
            manifests=loaded,
        )

    def _normalize_expected_fingerprints(
        self,
        expected_fingerprints: Mapping[str, str] | None,
        required: frozenset[str],
    ) -> dict[str, str]:
        if expected_fingerprints is None:
            return {}
        unknown = set(expected_fingerprints) - set(self._nodes)
        if unknown:
            raise StageGraphContractError(f"expected fingerprints contain unknown stages: {sorted(unknown)}")
        normalized: dict[str, str] = {}
        for stage, value in expected_fingerprints.items():
            if not _SHA256_PATTERN.fullmatch(str(value)):
                raise StageGraphContractError(f"expected fingerprint for {stage} is not a SHA-256 digest")
            if stage in required:
                normalized[stage] = str(value)
        return normalized

    def _validate_outputs(
        self,
        manifest: StageManifest,
        issues: list[StageValidationIssue],
    ) -> None:
        for expected in manifest.output_file_sha256:
            try:
                actual = describe_file(self.item_root, expected.path)
            except ArtifactContractError as exc:
                issues.append(
                    StageValidationIssue(
                        code="output_missing",
                        stage_name=manifest.stage_name,
                        path=expected.path,
                        detail=str(exc),
                    )
                )
                continue
            if actual != expected:
                issues.append(
                    StageValidationIssue(
                        code="output_digest_mismatch",
                        stage_name=manifest.stage_name,
                        path=expected.path,
                        detail="declared output size or SHA-256 differs from disk",
                    )
                )

    def _validate_upstream_marker_digests(
        self,
        required: frozenset[str],
        manifests: Mapping[str, StageManifest],
        issues: list[StageValidationIssue],
    ) -> None:
        for stage in self._topological_order:
            if stage not in required or stage not in manifests:
                continue
            manifest = manifests[stage]
            for upstream_stage, expected_digest in manifest.upstream_manifests:
                if upstream_stage not in required:
                    continue
                upstream_relative = manifest_relative_path(upstream_stage)
                upstream_marker = self.item_root / Path(*upstream_relative.split("/"))
                if not upstream_marker.is_file():
                    continue
                try:
                    actual_digest = sha256_file(upstream_marker)
                except ArtifactContractError:
                    continue
                if actual_digest != expected_digest:
                    issues.append(
                        StageValidationIssue(
                            code="upstream_manifest_mismatch",
                            stage_name=stage,
                            path=upstream_relative,
                            detail=f"stored {upstream_stage} marker digest differs from disk",
                        )
                    )

    def _validate_output_ownership(
        self,
        manifests: Mapping[str, StageManifest],
        issues: list[StageValidationIssue],
    ) -> None:
        owners: dict[str, list[str]] = {}
        marker_paths = {manifest_relative_path(stage) for stage in VALID_STAGE_NAMES}
        for stage, manifest in manifests.items():
            for output in manifest.output_file_sha256:
                owners.setdefault(output.path, []).append(stage)
                if output.path in marker_paths:
                    issues.append(
                        StageValidationIssue(
                            code="stage_marker_owned_as_output",
                            stage_name=stage,
                            path=output.path,
                            detail="a stage payload cannot own any stage commit marker",
                        )
                    )
        for path, stages in owners.items():
            unique_stages = sorted(set(stages), key=lambda stage: self._order_index.get(stage, 999))
            if len(unique_stages) > 1:
                issues.append(
                    StageValidationIssue(
                        code="output_ownership_conflict",
                        stage_name=unique_stages[0],
                        path=path,
                        detail=f"path is declared by stages {unique_stages}",
                    )
                )

    def _issue_sort_key(self, issue: StageValidationIssue) -> tuple[int, str, str, str]:
        return (
            self._order_index.get(issue.stage_name, 999),
            issue.code,
            issue.path or "",
            issue.detail,
        )
