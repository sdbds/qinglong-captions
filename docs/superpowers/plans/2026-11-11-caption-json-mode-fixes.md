# Caption JSON and Mode Fixes Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement these tasks with test-first verification.

**Goal:** Fix issues #22 and #23 without changing user prompts, retry policy, unrelated metadata, or existing non-image output contracts.

**Architecture:** Satisfy JSON-mode preconditions at the OpenAI-compatible request boundary. Normalize explicitly sectioned caption strings into canonical fields before mode filtering, sharing this logic between the provider and image postprocessing. Keep output writers unchanged so all persistence paths consume the same payload.

**Tech Stack:** Python 3.11, pytest, the installed OpenAI SDK, httpx MockTransport.

**Spec:** https://github.com/sdbds/qinglong-captions/issues/22 and https://github.com/sdbds/qinglong-captions/issues/23 (read in this chat).

## Global Constraints

- Preserve all pre-existing worktree edits; do not upgrade dependencies or call paid model endpoints.
- No push, PR creation, or issue closure in this task. Leave changes uncommitted for inspection.
- Existing explicit JSON instructions/schema and plaintext mode must remain unchanged.
- Only explicit JSON-format incompatibility triggers plaintext fallback; transport failures retain unified retries.
- Recognize only explicit short/long section headings in top-level caption fields, never recursively rewrite arbitrary JSON.
- Preserve plain captions, nested custom data, scores, tags, failed/skipped results, and non-image postprocessing.

## Review Focus

- Media data URLs containing `json` must not suppress the text instruction.
- Format fallback must restore original messages without mutating reusable prompts.
- Existing explicit canonical fields must win over aliases and embedded sections.
- Incomplete or ambiguous headings must not silently discard caption text.
- Fenced custom JSON without an active template must retain its existing contract.

## Task 1: JSON Request Contract (#22)

**Files:** `module/providers/cloud_vlm/openai_compatible.py`, `tests/test_provider_retry_regressions.py`.

**Interface:** `_ensure_json_instruction(messages: list[dict]) -> list[dict]` returns unchanged messages when their text mentions JSON, otherwise copies the message containers it changes and adds a valid-object instruction, defaulting to a `description` string when no schema is requested.

- [ ] Add an offline endpoint validator and test ordinary/Chinese prompts, empty system, existing case-insensitive JSON instructions, image/pair/video inputs, plaintext mode, fallback, and immutability.
- [ ] Run the new tests and observe the missing-JSON validation failure before changing production code.
- [ ] Apply the helper only in JSON mode; restore original messages with format fallback.
- [ ] Run ` .venv/Scripts/python.exe -m pytest tests/test_provider_retry_regressions.py tests/test_provider_exception_logging.py -q` and require passing results.

## Task 2: Structured Caption Modes (#23)

**Files:** create `utils/caption_modes.py` and `tests/test_caption_modes.py`; modify `module/providers/cloud_vlm/openai_compatible.py`, `module/caption_pipeline/postprocess.py`.

**Interface:** `normalize_caption_modes(payload: dict, mode: str) -> dict` shallow-copies and normalizes recognized top-level caption fields, then applies the requested mode without mutating nested metadata. The provider preserves its legacy plain-description alias behavior independently.

- [ ] Test the exact embedded `###Short:` / `###Long:` example through provider filtering and postprocessing, for dict, JSON, CaptionResult, and fenced JSON inputs, across long/short/all.
- [ ] Assert literal expected canonical fields, TXT contents, JSON sidecars, and dataset serialization. Check field precedence, case/heading variants, unchanged ordinary/custom data, and non-image output.
- [ ] Observe these tests fail before adding the shared normalization helper and using it at structured-image return paths.
- [ ] Run ` .venv/Scripts/python.exe -m pytest tests/test_caption_modes.py tests/test_caption_result_pipeline.py tests/test_image_prompt_template.py tests/test_output_writer.py -q` and require passing results.

## Final Verification

- [ ] Review only this task's diff with a fresh reviewer.
- [ ] Run the project's authoritative offline suite: `.venv/Scripts/python.exe -m pytest tests -q --strict-markers -m "not optional_runtime and not gpu and not network" --durations=25`.
- [ ] Run targeted static checks and `git diff --check`; report unrelated pre-existing failures without modifying them.

## Execution Notes

- Initial worktree has many unrelated changes, but the implementation targets above are clean. Work in place to preserve the user's configured runtime and edits rather than duplicate the environment or change its dependency set.
- Existing no-template fenced custom JSON remains unparsed; only recognized caption-bearing fenced JSON is eligible for the new normalization path.
