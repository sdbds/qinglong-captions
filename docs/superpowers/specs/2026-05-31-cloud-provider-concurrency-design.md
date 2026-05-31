# Cloud Provider Bounded Concurrency Design

Date: 2026-05-31

## Problem

Current caption scheduling is effectively serial:

- `module/caption_pipeline/orchestrator.py` scans Lance with `batch_size=1`.
- `process_batch()` iterates each row and calls `api_process_batch_fn()` synchronously.
- `_process_segmented_media()` splits long media, then calls the provider once per chunk in a serial loop.
- `--codex_max_concurrency` exists in CLI / GUI config, but it is not consumed by the scheduler or Codex client layer.

This means cloud image caption providers are used like a single-lane queue even when the selected provider and account can safely handle multiple in-flight requests.

## First Principles

The bottleneck for cloud image captioning is mostly remote latency, not local CPU. Independent image rows do not need to share state. Therefore bounded concurrency can improve throughput by overlapping remote waits.

The hard boundary is correctness:

- output must stay attached to the original file;
- dataset updates must preserve URI/result pairing;
- local preprocessing must not be multiplied accidentally;
- provider/account limits must not be exceeded by default;
- stateful cloud providers such as Codex must not share mutable thread state across concurrent jobs.

So the target is not "high concurrency". The target is opt-in, bounded cloud image concurrency with deterministic outputs.

## Goals

1. Add scheduler support for concurrent cloud image provider calls.
2. Include both API-key cloud providers and `codex_subscription`.
3. Keep default behavior serial: concurrency defaults to `1`.
4. Preserve deterministic `processed_filepaths` / `results` ordering.
5. Keep local in-process VLM / ALM / OCR providers serial for this change.
6. Keep video, audio, PDF/document, and segmented-media paths serial in the first implementation.
7. Keep Lance dataset writes and final export single-threaded.
8. Make provider concurrency eligibility an explicit provider capability, not scheduler hardcoding.

## Non-Goals

- No GPU/local model parallelism.
- No video/audio/PDF/document concurrency in the first implementation.
- No segmented chunk concurrency in the first implementation.
- No adaptive rate-limit controller in the first implementation.
- No concurrent Lance `merge_insert`.
- No change to provider prompt semantics.
- No automatic "best" concurrency detection.

## Terminology

Cloud provider:

- Remote provider backed by a network service.
- Includes API-key providers such as Gemini, Kimi, QwenVL, GLM, Ark, MiniMax, MiMo, Mistral/Pixtral, OpenAI-compatible, StepFun.
- Includes `codex_subscription`, even when it uses ChatGPT/Codex subscription auth instead of an API key.

Local provider:

- In-process CPU/GPU provider using local model weights.
- Stays serial in this design.

Caption job:

- One image dataset row in `process_batch()`.
- Text/video/audio/PDF/document rows remain serial in phase 1.

## Why Phase 1 Is Image-Only

The provider API call is not the only work inside one scheduler job. The current per-file body also creates scene detectors, may start background scene detection, and may run ffmpeg splitting for long media.

If the first implementation runs every cloud-capable row concurrently, `cloud_max_concurrency=4` on a video dataset can also start four scene detectors and four ffmpeg split jobs at once. That is local CPU/IO concurrency, not just cloud API concurrency.

Phase 1 therefore applies cloud concurrency only to image MIME rows. This covers the existing production image caption path without accidentally multiplying local preprocessing cost.

`text/*` rows are not included in phase 1 because there is no production caption provider route for them today.

`application/*` documents such as PDF, DOCX, PPTX, and EPUB can trigger OCR, page rendering, or document parsing side effects and remain serial.

Future work can add separate local preprocessing limits for video/audio/PDF/document paths.

## CLI / Config Surface

Add a global cloud concurrency option:

```text
--cloud_max_concurrency=<int>
```

Rules:

- Default: `1`.
- Minimum accepted value: `1`.
- Values above `1` only affect image rows handled by providers whose capability declares cloud concurrency support.
- Local providers ignore it and run serial.
- Video, audio, PDF/document, and segmented media ignore it in phase 1 and run serial.

Keep existing Codex-specific option:

```text
--codex_max_concurrency=<int>
```

Rules:

- Default: `1`.
- Applies only to `codex_subscription`.
- Effective Codex concurrency is `min(cloud_max_concurrency, codex_max_concurrency)`.
- This makes the existing GUI/CLI field meaningful without letting Codex accidentally inherit a high global cloud value.

PowerShell launcher additions:

```powershell
$cloud_max_concurrency = 1
$codex_max_concurrency = 1
```

GUI:

- Keep the existing Codex max concurrency input.
- The existing Codex input is gated by `cloud_max_concurrency`; Codex concurrency requires both values to be greater than `1`.
- Add the global cloud concurrency GUI control after CLI/PowerShell scheduling is implemented and tested.
- Do not expose a GUI knob before it is wired end to end.

Rejected name:

- `api_max_concurrency` is too narrow because `codex_subscription` is cloud-backed but not always API-key-backed.

## Provider Capability Contract

Extend `ProviderCapabilities` with one field only:

```python
@dataclass
class ProviderCapabilities:
    supports_streaming: bool = False
    supports_structured_output: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    supports_images: bool = False
    supports_documents: bool = False
    max_file_size_mb: int = 100
    supported_mimes: Optional[List[str]] = None

    supports_cloud_concurrency: bool = False
```

Defaults are conservative. Providers must opt in.

Base-class defaults:

- `CloudVLMProvider`: `supports_cloud_concurrency=True` after its image call paths are audited.
- `VisionAPIProvider`: `supports_cloud_concurrency=True` after its image call paths are audited.
- `OCRProvider`: keep `supports_cloud_concurrency=False` by default because most OCR providers are local in-process models.

Explicit provider handling:

- `codex_subscription`: `supports_cloud_concurrency=True`, but still uses `codex_max_concurrency` as an additional cap.
- Remote OCR providers that are genuinely API-backed can opt in individually. `mistral_ocr` inherits `VisionAPIProvider`, so it is covered. `qianfan_ocr` should opt in only if its current implementation is remote-service-only and has no shared mutable local model state.

The scheduler must not maintain a provider-name allowlist except as a test fixture.

Do not add `cloud_concurrency_group` or `default_cloud_max_concurrency` in phase 1. They are premature and create bad default-value semantics.

## Effective Concurrency

Keep the calculation simple:

```text
if not job.mime.startswith("image/"):
    effective = 1
elif not provider.capabilities.supports_cloud_concurrency:
    effective = 1
elif provider.name == "codex_subscription":
    effective = min(args.cloud_max_concurrency, args.codex_max_concurrency)
else:
    effective = args.cloud_max_concurrency
```

There is no provider default cap in phase 1. The global knob is the cap for normal cloud providers. Codex keeps its existing dedicated cap.

## Scheduler Shape

Introduce small data objects in `module/caption_pipeline/orchestrator.py` or a new `module/caption_pipeline/scheduler.py`:

```python
@dataclass(frozen=True)
class CaptionJob:
    index: int
    filepath: str
    mime: str
    duration: int
    sha256hash: str

@dataclass
class CaptionJobResult:
    index: int
    filepath: str
    mime: str
    output: Any
    log_text: str = ""
```

Extract the current per-file body from `process_batch()` into:

```python
def process_single_caption_job(job, args, config, api_process_batch_fn, console_obj) -> CaptionJobResult:
    ...
```

This function owns the current per-row behavior:

- scene detector creation for serial video/audio paths;
- segment-time resolution;
- direct vs segmented processing;
- provider call;
- postprocess;
- subtitle scene alignment;
- sidecar caption write for successful jobs.

It does not own:

- Lance dataset update;
- final Lance export;
- final ordered result aggregation.

## Execution Model

Use `ThreadPoolExecutor`, not asyncio.

Reason:

- Current providers are synchronous blocking clients.
- Rewriting all provider SDK calls to async would increase blast radius.
- Remote API waits release the thread to the OS and are a good fit for bounded worker threads.

Main algorithm:

1. Read dataset rows into `CaptionJob` records with stable indexes.
2. Resolve the selected provider once for the current args/mime class where possible.
3. Split rows into phase-1 concurrent image jobs and serial jobs.
4. If effective concurrency is `1`, run the existing serial behavior through the extracted single-job function.
5. If effective concurrency is greater than `1`, submit eligible image jobs to a bounded `ThreadPoolExecutor`.
6. Store every result by `job.index`.
7. Run non-eligible rows serially through the same single-job function.
8. Call `update_dataset_captions()` once with ordered `processed_filepaths` and `results`.
9. Call `extract_from_lance_fn()` once after dataset update.

Ordering rule:

- Completion order never determines output order.
- `job.index` determines final `processed_filepaths` and `results` order.

## Sidecar Write Policy

Keep current behavior: successful jobs write their own sidecar immediately.

Reason:

- Each job writes a distinct file.
- There is no shared sidecar write target.
- Users currently get partial useful sidecar output if a later item fails.
- Lance dataset update remains all-or-nothing at the batch level because it is still called after job collection.

If all-or-nothing sidecar writes are desired later, add an explicit option. Do not make it an incidental behavior change in the concurrency refactor.

## Logging and Progress

Do not share Rich live progress objects across worker threads.

Worker rules in concurrent mode:

- Workers get `progress=None`.
- Workers write logs to a per-job buffered console, such as `Console(file=io.StringIO(), force_terminal=False)`.
- The main thread flushes buffered logs after each job completes or after ordered completion.

Main-thread progress:

- One progress task tracks total completed jobs.
- The main thread advances it as futures complete.
- Provider streaming output is suppressed or buffered in concurrent mode to avoid interleaved terminal output.

Implementation checklist:

- Audit every cloud provider call path for unconditional `console.print()` streaming.
- Ensure concurrent mode passes a buffered console to the provider context.
- Accept that live token streaming is a serial-mode feature unless a provider explicitly supports thread-safe structured streaming.

This is a deliberate tradeoff: concurrent mode favors throughput and correct logs over live token streaming.

## Provider Thread-Safety Audit

API-key provider concurrency depends on a simple property: a job must not share mutable request state with another job.

Known good foundation:

- `api_handler_v2.api_process_batch()` creates a new provider instance per request.
- `ProviderContext` is also created per request.
- Many provider clients are created inside `attempt()`.

Implementation must still audit these before enabling each provider:

- no module-level shared mutable client with request-local state;
- no shared output file path;
- no shared mutable prompt/result cache without locking;
- streaming output can be suppressed or buffered;
- retry sleeps are worker-local.

Any provider that fails the audit keeps `supports_cloud_concurrency=False` until fixed.

## Codex Subscription Requirements

Codex must be treated as cloud, but not as a stateless HTTP provider.

Current risk:

- `CodexAppServerCaptionClient` stores mutable `self.thread_id` and `self.thread`.
- `_CLIENT_CACHE` can return the same client to concurrent callers.
- Concurrent `caption_image()` calls can race and return wrong thread metadata or run turns against the wrong mutable thread object.

Required changes before `codex_max_concurrency > 1` is honored:

1. Make app-server caption calls thread-safe.
2. Remove request-specific mutable state from `CodexAppServerCaptionClient`, or isolate it per worker slot.
3. Protect `_CLIENT_CACHE` with a lock.
4. Protect `ensure_auth()` with a lock or make it idempotent under concurrent calls.
5. Keep one ephemeral thread per image job to avoid context pollution.

Recommended implementation:

- Add a small Codex client pool keyed by config and slot id.
- Pool size is `codex_max_concurrency`.
- Each worker acquires one slot, uses that slot's client for one request, then releases it.
- Each slot has at most one in-flight `caption_image()`.
- This avoids assuming the underlying SDK client is fully thread-safe.

For `codex_backend="exec"`:

- Each worker can run a separate subprocess.
- Still apply `codex_max_concurrency` because subprocess startup, subscription limits, and auth state are shared external resources.

For `codex_backend="sdk_app_server"`:

- Prefer the slot-pool approach first.
- A later optimization may share one app-server client if the SDK explicitly guarantees concurrent thread/turn calls.

## Segmented Media

Segmented media stays serial in phase 1.

Reason:

- Split and scene-alignment work are local CPU/IO work.
- Chunk merge order and temporary file cleanup add extra edge cases.
- Text/image row concurrency gives the highest value with the smallest behavioral surface.

Future design:

- Add separate local preprocessing limits.
- Then consider processing cloud-capable chunks concurrently.
- Sort chunk results by chunk index before merging subtitles or structured segment summaries.

## Failure Semantics

Serial behavior currently stops on the first uncaught provider exception after any already-completed sidecars have been written.

Concurrent behavior should preserve that practical contract:

- Stop scheduling new jobs after the first failure.
- Let already-running futures finish or cancel pending futures where possible.
- Raise the first failure with provider/job context.
- Do not call `update_dataset_captions()` if the batch did not complete.
- Keep sidecars already written by successful jobs.

## Rate Limit Behavior

Keep existing retry behavior in phase 1:

- Worker-local retries call `with_retry_impl()`.
- A 429 blocks only that worker, not the entire batch.
- Other workers may continue.

Known limitation:

- If every worker hits 429, the system will wait in parallel and retry in parallel, which may cause repeated bursts.

Future improvement:

- Add an adaptive provider-specific dampener after baseline concurrency is correct.
- That is out of scope for the first implementation.

## Tests

Add focused tests before implementation.

Scheduler tests:

1. `test_process_batch_cloud_image_jobs_run_concurrently`
   - Fake cloud image provider sleeps.
   - `cloud_max_concurrency=3`.
   - Assert max in-flight calls is greater than `1`.

2. `test_process_batch_preserves_order_under_concurrency`
   - Fake provider returns results out of completion order.
   - Assert dataset update receives results in original URI order.

3. `test_process_batch_local_provider_remains_serial`
   - Fake local provider capability has `supports_cloud_concurrency=False`.
   - Assert max in-flight calls is `1` even when `cloud_max_concurrency=4`.

4. `test_process_batch_video_cloud_provider_remains_serial_in_phase_1`
   - Fake cloud provider can handle video.
   - Video jobs still run with max in-flight `1`.

5. `test_process_batch_application_document_remains_serial_in_phase_1`
   - Fake cloud provider can handle `application/pdf`.
   - Document jobs still run with max in-flight `1`.

6. `test_concurrent_failure_skips_dataset_update_but_keeps_successful_sidecars`
   - One fake job fails after another succeeds.
   - Assert `update_dataset_captions()` is not called.
   - Assert successful sidecar write occurred.
   - Assert first error includes filepath/provider context.

7. `test_retry_sleep_blocks_only_one_worker`
   - One fake job sleeps/retries.
   - Another fake job completes.
   - Assert another worker can finish while first is sleeping.

Codex tests:

8. `test_codex_max_concurrency_limits_in_flight_calls`
   - Fake Codex client records max in-flight.
   - `cloud_max_concurrency=4`, `codex_max_concurrency=2`.
   - Assert max in-flight is `2`.

9. `test_codex_app_server_pool_uses_distinct_slots`
   - Fake SDK client exposes slot id.
   - Concurrent calls should not share request-specific `thread_id/thread` state.

10. `test_codex_default_remains_serial`
   - `cloud_max_concurrency=4`, default `codex_max_concurrency=1`.
   - Assert Codex in-flight max is `1`.

CLI / GUI tests:

11. Parser accepts `--cloud_max_concurrency`.
12. PowerShell launcher passes `--cloud_max_concurrency` only when value is greater than `1`.
13. Existing `--codex_max_concurrency` remains accepted and now affects scheduler behavior.
14. GUI global `cloud_max_concurrency` test is added only when the GUI control is implemented.

## Implementation Plan

1. Add `cloud_max_concurrency` to CLI parser, PowerShell launchers, and tests.
2. Extend `ProviderCapabilities` with `supports_cloud_concurrency`.
3. Mark audited cloud base classes and `codex_subscription`.
4. Extract single-job execution from `process_batch()`.
5. Add a bounded thread scheduler for eligible image rows.
6. Keep rows outside `image/*` serial in phase 1.
7. Preserve ordered aggregation and single-threaded Lance update/export.
8. Keep successful sidecar writes inside single-job execution.
9. Add buffered worker logging and suppress interleaved streaming output in concurrent mode.
10. Add Codex client slot pool or equivalent per-slot isolation.
11. Wire `codex_max_concurrency` into effective Codex concurrency.
12. Defer segmented-media chunk concurrency until the phase-1 scheduler is stable.

## Review Decisions

1. Global option name: use `cloud_max_concurrency`, not `api_max_concurrency`.
2. Sidecar write timing: keep current successful-job-immediate behavior.
3. Codex app-server: one client per concurrency slot is acceptable; correctness is more important than startup overhead.
4. GUI: CLI/PowerShell first; add GUI global control after the scheduler is wired and tested.

## Recommendation

Use `cloud_max_concurrency` as the global knob, keep it defaulted to `1`, and make every provider opt in through `supports_cloud_concurrency`.

Implement image row concurrency first. Do not start with segment-level concurrency, video/audio/PDF/document/text concurrency, provider groups, default provider caps, or adaptive rate limiting. The smaller first cut proves ordering, failure behavior, logging, and Codex isolation without turning the scheduler into a rate-control system.
