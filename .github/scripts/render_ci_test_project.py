from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def build_ci_test_project_text(source_text: str) -> str:
    data = tomllib.loads(source_text)
    project = data.get("project")
    groups = data.get("dependency-groups")

    if not isinstance(project, dict):
        raise TypeError("pyproject.toml is missing [project]")
    if not isinstance(groups, dict):
        raise TypeError("pyproject.toml is missing [dependency-groups]")

    name = project.get("name")
    version = project.get("version")
    requires_python = project.get("requires-python")
    dependencies = project.get("dependencies", [])
    test_dependencies = groups.get("test")

    if not isinstance(name, str):
        raise TypeError("[project].name must be a string")
    if not isinstance(version, str):
        raise TypeError("[project].version must be a string")
    if not isinstance(requires_python, str):
        raise TypeError("[project].requires-python must be a string")
    if not isinstance(dependencies, list):
        raise TypeError("[project].dependencies must be a list")
    if not isinstance(test_dependencies, list):
        raise TypeError("[dependency-groups].test must be a list")

    lines = [
        "[project]",
        f'name = {json.dumps(name + "-ci-test", ensure_ascii=False)}',
        f"version = {json.dumps(version, ensure_ascii=False)}",
        f'requires-python = {json.dumps(requires_python, ensure_ascii=False)}',
        f"dependencies = {json.dumps(dependencies, ensure_ascii=False)}",
        "",
        "[dependency-groups]",
        f"test = {json.dumps(test_dependencies, ensure_ascii=False)}",
        "",
        "[tool.uv]",
        "package = false",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a minimal pyproject.toml for CI test jobs without resolving optional extras."
    )
    parser.add_argument("--source", required=True, help="Path to the source pyproject.toml")
    parser.add_argument("--output", required=True, help="Path to the rendered minimal pyproject.toml")
    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)
    rendered = build_ci_test_project_text(source_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
