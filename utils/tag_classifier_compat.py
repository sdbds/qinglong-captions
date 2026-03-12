from __future__ import annotations


class _FallbackTagClassifier:
    def classify(self, tags: list[str]) -> dict[str, list[str]]:
        return {"0": list(tags)}

    def get_colored_tag(self, tag: str) -> str:
        return tag


try:
    from utils.tag_highlighting import get_tag_classifier as _get_tag_classifier
except ModuleNotFoundError as exc:
    if exc.name != "utils.tag_highlighting":
        raise

    def get_tag_classifier() -> _FallbackTagClassifier:
        return _FallbackTagClassifier()

else:
    get_tag_classifier = _get_tag_classifier
