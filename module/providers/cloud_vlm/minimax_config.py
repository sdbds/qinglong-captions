"""Request policy shared by the MiniMax API and Code providers."""

from typing import Any

DEFAULT_MINIMAX_MODEL_ID = "MiniMax-M3.1-Flash"
DEFAULT_MINIMAX_REASONING_EFFORT = "medium"
DEFAULT_MINIMAX_MAX_COMPLETION_TOKENS = 8192
_M31_MODEL = DEFAULT_MINIMAX_MODEL_ID.casefold()
_M31_PREVIEW_MODEL = "minimax-m3.1-flash-preview"
_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


class MiniMaxRequestError(ValueError):
    """A request needs configuration changes, not an identical paid retry."""


def minimax_request_options(
    model_path: str,
    *,
    reasoning_split: bool,
    reasoning_effort: str | None = None,
    max_completion_tokens: int = DEFAULT_MINIMAX_MAX_COMPLETION_TOKENS,
    code_mode: bool = False,
) -> dict[str, Any]:
    model = model_path.strip().casefold()
    if isinstance(max_completion_tokens, bool) or not isinstance(max_completion_tokens, int) or max_completion_tokens <= 0:
        raise MiniMaxRequestError("MiniMax max_completion_tokens must be a positive integer")
    if model in {_M31_MODEL, _M31_PREVIEW_MODEL}:
        reasoning_split = True
    options: dict[str, Any] = {
        "extra_body": {"reasoning_split": reasoning_split} if code_mode or reasoning_split else None,
    }
    if model == _M31_MODEL:
        effort = DEFAULT_MINIMAX_REASONING_EFFORT if reasoning_effort is None else str(reasoning_effort).strip().lower()
        if effort not in _EFFORTS:
            raise MiniMaxRequestError(
                f"Unsupported MiniMax reasoning effort: {reasoning_effort}. Expected low, medium, high, xhigh, max"
            )
        options["reasoning_effort"] = effort
    if model in {_M31_MODEL, "minimax-m3"}:
        options["max_completion_tokens"] = max_completion_tokens
    else:
        options.update(
            max_tokens=max_completion_tokens,
            temperature=0.3 if code_mode and reasoning_split else 0.7,
            top_p=0.95,
        )
    return options


def validate_minimax_response(text: str, finish_reason: str | None) -> str:
    if finish_reason == "length":
        raise MiniMaxRequestError("MiniMax response truncated; increase max_completion_tokens or lower reasoning_effort")
    if not text.strip():
        raise RuntimeError("RETRY_EMPTY_CONTENT: MiniMax returned no answer content")
    return text
