# MuSViT ONNX Full-Page OMR Inference Design

## Status

This design replaces the user-facing contract in
`2026-07-02-musvit-onnx-sheet-music-tool-design.md`.

The old tool extracts MuSViT encoder embeddings and writes `embedding.npz`.
That behavior does not solve Optical Music Recognition. The sheet-music tool
will now accept score images, PDFs, or directories and produce MusicXML and/or
MIDI through the fine-tuned full-page OMR model:

`https://huggingface.co/bdsqlsz/qinglong-musvit-1.0`

Embedding extraction is not retained as a GUI mode, fallback, or alternate
output. This spec changes the existing sheet-music feature in place. It does
not add training or checkpoint export.

Reviewed on 2026-07-23 against:

- model revision `6f47cefe0e736fbdd0eab9e8bc8d4602e81939fe`;
- reference branch
  `https://github.com/sdbds/MuSViT/tree/codex/full-page-omr-throughput`;
- reference branch commit `c3760a940ba8800c16dc141c178bf63448bb614d`.

## Linus Review

### Is this a real problem?

Yes. Encoder embeddings are not symbolic music and cannot be converted into a
score without a trained downstream OMR decoder. The new repository contains
that decoder and therefore closes the missing product path.

### Is there a simpler way?

Yes. Consume the uploaded two-graph ONNX bundle directly:

1. load `encoder.onnx` and `decoder.onnx` through the project's shared ONNX
   runtime;
2. encode each page once;
3. run the validated host-side greedy decoder loop;
4. reconstruct standard Humdrum `**kern`;
5. parse the notation and export MusicXML/MIDI.

Do not vendor the PyTorch training model, reconstruct a Lightning checkpoint,
call the upstream training CLI, or re-export weights inside this project.

### What breaks?

This is an intentional replacement of the current sheet-music contract:

- `.npz` embedding output disappears;
- `batch_size` and `preprocess_mode` disappear because the OMR graphs have
  fixed batch and preprocessing contracts;
- the default model changes from `bdsqlsz/musvit-onnx` to
  `bdsqlsz/qinglong-musvit-1.0`;
- existing embedding CLI flags, GUI labels, docs, and tests must be replaced;
- existing embedding output directories are left untouched but are never
  treated as completed OMR work.

The proposal is approved with the following corrections incorporated:

1. the uploaded model does not contain the `metadata.json` required by the
   reference runtime, so that runtime cannot be copied unchanged;
2. MuScriptor's audio-derived score model is not a valid direct input for OMR.
   Only neutral export orchestration and score writers may be shared;
3. the published `image_size` and bilinear preprocessing contract is strict;
   the legacy permissive loader and bicubic helper are forbidden;
4. Kern header and terminator reconstruction is an exact two-spine transform,
   not a heuristic repair;
5. `module.music_export` becomes the single owner now, and the 2026-07-19
   enriched-export design and plan are revised in the same change;
6. MuSViT generation ids and limits come from `w2i` and custom `maxlen`, never
   generic Hugging Face generation fields;
7. the revision reaches downloads and repository listing, while provider
   fallback and preprocessing parity receive explicit regression tests.

## Verified Model Contract

### Repository Contents

The model repository is public and ungated at the reviewed revision. It
contains:

```text
README.md
config.json
encoder.onnx
decoder.onnx
preprocessor_config.json
val_evaluation.json
```

It does not contain `metadata.json`.

The model card currently labels the repository as `diffusers` and
`image-text-to-text`. Those tags are not a runtime API. The project must treat
the repository as a generic ONNX bundle and must not use a Diffusers or
Transformers auto-loader.

Artifact identities at the reviewed revision:

| File | Size | LFS SHA-256 |
| --- | ---: | --- |
| `encoder.onnx` | 356,197,419 bytes | `1d49c2c15cb91ce6d69f3b4591d0b0fb5e63ce81dcd6089da6d1fd0e53d530f0` |
| `decoder.onnx` | 29,469,910 bytes | `748f844cdd8ad2298cbc72579be751e8edca89df4f6fabfd7b0fa88032f37423` |

### Encoder Graph

Input:

| Name | Dtype | Shape |
| --- | --- | --- |
| `pixel_values` | float32 | `[1, 3, 1024, 1024]` |

Outputs:

| Name | Dtype | Shape |
| --- | --- | --- |
| `raw_features` | float32 | `[1, 4096, 256]` |
| `enhanced_features` | float32 | `[1, 4096, 256]` |

The encoder includes the ViT foundation model, CLS removal, adaptor, and 2D
positional preparation. No separate foundation weight download is required at
runtime.

### Decoder Graph

Inputs:

| Name | Dtype | Shape |
| --- | --- | --- |
| `raw_features` | float32 | `[1, 4096, 256]` |
| `enhanced_features` | float32 | `[1, 4096, 256]` |
| `token_ids` | int64 | `[1, T]` |

Output:

| Name | Dtype | Shape |
| --- | --- | --- |
| `next_token_logits` | float32 | `[1, 215]` |

Only the prefix length `T` is dynamic. Batch size and page resolution are
fixed.

### Generation Contract

The bundled config declares:

```text
architecture       SMT
foundation         ViTMAEBase
foundation weights carlospm12/LSMT-MAE-Base-1024-16
decoder dimension  256
decoder layers     8
vocabulary size    215
maximum length     7512
padding id         0
BOS id             100
EOS id             183
space id           44
tab id             29
line-break id      132
```

The runtime performs uncached full-prefix greedy decoding:

1. run the encoder once for the page;
2. initialize the prefix with BOS id `100`;
3. run `decoder.onnx` with the complete prefix;
4. append `argmax(next_token_logits)`;
5. stop at EOS id `183` or the effective token limit;
6. map ids through `config.json.i2w`.

The reference branch explicitly rejected the experimental KV-cache path after
it produced a token divergence on one validation page. The first project
integration therefore preserves the full-prefix graph contract. It does not
introduce beam search, cache emulation, repetition penalties, or ONNX `Loop`.

### Preprocessing Contract

The loader accepts exactly the published `preprocessor_config.json` schema. Its
top-level key set must be:

```text
color
do_normalize
do_rescale
do_resize
image_size
input_layout
interpolation
rescale_factor
```

Missing and unknown keys fail bundle validation. Required values are:

```text
color             "RGB"
do_normalize      false
do_rescale        true
do_resize         true
image_size        [1024, 1024]
input_layout      "NCHW"
interpolation     "bilinear"
rescale_factor    0.00392156862745098
```

The OMR tensor path is one fixed algorithm:

1. convert the page to RGB;
2. resize directly to `1024 x 1024` with
   `Image.Resampling.BILINEAR`;
3. materialize an HWC `numpy.float32` array;
4. divide by `numpy.float32(255.0)`;
5. transpose to contiguous NCHW;
6. add the fixed batch dimension, producing `[1, 3, 1024, 1024]`;
7. do not apply mean/std normalization.

The OMR implementation must not call the existing
`load_musvit_preprocessor_config`, `_as_size_tuple`, or
`preprocess_pil_image` helpers in `module.sheet_music_musvit`. Those helpers
parse the legacy `size` key permissively and hard-code bicubic interpolation,
so reusing them would silently diverge from the published validation pipeline.

The old `page_resize` versus `pad_square` choice is removed. The model has one
validated preprocessing path.

### Published Evaluation Evidence

`val_evaluation.json` records a ten-page Polish Scores validation run:

```text
CER_v2                 8.7692
SER_v2                10.3882
LER_v2                27.5690
mean elapsed/page     21.5614 seconds
EOS pages             10/10
truncated pages        0/10
```

This is evidence that the uploaded graphs run and match the rebuilt evaluation
bundle. It is not a claim of general score accuracy. The model is fine-tuned on
the Polish Scores distribution and should not be presented as a universal
handwritten, orchestral, or arbitrary-layout OMR model.

## Goals

1. Accept supported image files, PDF files, and directories.
2. Stream PDF pages through `utils.stream_util.iter_pdf_pages_high_quality`.
3. Download and load the pinned two-graph ONNX bundle once per process.
4. Encode each page once and decode one symbolically constrained greedy
   sequence.
5. Preserve raw token ids and strings for diagnosis.
6. Reconstruct a standard Humdrum `**kern` document.
7. Export requested MusicXML, MIDI, or both.
8. Combine every successful page of one PDF into one ordered MusicXML/MIDI
   score.
9. Produce deterministic per-page, per-document, and job metadata.
10. Continue other pages after a page-level recognition or conversion failure.
11. Remove embedding output from the normal sheet-music code path.

## Non-Goals

- No training, fine-tuning, checkpoint conversion, or ONNX export.
- No PyTorch, Transformers, Lightning, W&B, or `datasets` runtime.
- No object detection or note-box output.
- No embedding output or silent fallback to the old encoder model.
- No KV-cache decoder until a separately validated graph has exact token
  parity.
- No TensorRT or FP16 path in the first integration.
- No heuristic correction of pitches, rhythms, barlines, or voices.
- No post-hoc insertion of guessed pitches, visible rests, barlines, tabs, or
  spine fields into a decoded page.
- No best-effort PDF result that silently omits failed pages.
- No claim that failed or truncated model output is valid MusicXML/MIDI.

## Architecture

### Entry Point

Keep the existing process-runner key and CLI module:

```text
module.sheet_music_musvit
```

Replace its embedding implementation with a thin OMR CLI and orchestration
layer. Do not add a compatibility wrapper around the old behavior.

Suggested units:

```text
module/sheet_music_musvit.py
module/sheet_music_omr/
  __init__.py
  model.py
  preprocess.py
  decode.py
  inputs.py
  pipeline.py
module/music_export/
  __init__.py
  service.py
  music21_writers.py
  validation.py
```

Responsibilities:

- `sheet_music_musvit.py`: arguments, config resolution, progress, exit code.
- `model.py`: model artifact contract, ONNX sessions, greedy generation.
- `preprocess.py`: strict published-config parsing and the single bilinear
  image-to-tensor path.
- `decode.py`: token mapping, BeKern reconstruction, `**kern` envelope and
  validation.
- `inputs.py`: deterministic image/PDF/directory page iteration.
- `pipeline.py`: page lifecycle, outputs, metadata, resume, manifest.
- `music_export.service`: neutral export jobs, temporary targets, validation
  callbacks, atomic replace, and per-format status.
- `music21_writers.py`: write a parsed notation score to MusicXML or MIDI.
- `validation.py`: re-open generated artifacts and reject invalid output.

### Shared ONNX Base

Use `module.onnx_runtime.multi_model.OnnxMultiModelSpec` with:

```text
artifacts:
  encoder -> encoder.onnx
  decoder -> decoder.onnx
support files:
  config -> config.json
  preprocessor_config -> preprocessor_config.json
  validation -> val_evaluation.json
```

Use the shared session configuration, provider selection, model download
logging, and session cache. Do not create a second ONNX runtime abstraction.

The shared artifact layer needs one real extension: optional Hugging Face
`revision` propagation for single-model and multi-model downloads. Existing
specs retain `revision=None`; the OMR model uses the pinned revision. Its local
directory must include the revision so stale files from another revision cannot
win through the current "existing target" fast path.

The same revision must be passed to both `hf_hub_download` and
`list_repo_files`. This includes every `list_repo_files` call used to discover
external ONNX data files; resolving the graph at one revision and its external
data at another is an invalid bundle. When a revision is present, the shared
artifact helpers must still delegate an existing target to `hf_hub_download`
so its revision metadata is checked; the unversioned existing-target shortcut
must not bypass revision resolution.

Example cache shape:

```text
huggingface/
  bdsqlsz_qinglong-musvit-1.0/
    6f47cefe0e736fbdd0eab9e8bc8d4602e81939fe/
      encoder.onnx
      decoder.onnx
      config.json
      preprocessor_config.json
      val_evaluation.json
```

### Bundle Validation

Because the Hub repository lacks `metadata.json`, validate the actual bundle
instead of fabricating a manifest:

1. require all five files;
2. for the default pinned revision, verify ONNX size and SHA-256 against the
   reviewed artifact identities;
3. validate graph input/output names, dtypes, and shapes after session load;
4. require `len(w2i) == len(i2w) == out_categories == 215`;
5. derive PAD, BOS, and EOS only from `w2i["<pad>"]`, `w2i["<bos>"]`, and
   `w2i["<eos>"]`, then require all six special tokens and ids recorded above;
6. derive the generation limit only from the custom `maxlen` field and require
   `maxlen == 7512`;
7. ignore generic Hugging Face generation fields such as `max_length`,
   `bos_token_id`, `eos_token_id`, `decoder_start_token_id`, and
   `pad_token_id`; at the pinned revision `max_length` is `20` and BOS/EOS are
   null, so these fields are not the MuSViT generation contract;
8. require the exact preprocessor key set, values, and bilinear tensor algorithm
   above;
9. fail before processing input when any invariant differs.

An explicitly overridden repository/revision cannot use the pinned hashes, but
it must pass every structural graph/config/preprocessor check.

### Runtime Provider

The validated graphs are FP32. The tool-specific default is CUDA with CPU
fallback:

```toml
[onnx_runtime.musvit]
execution_provider = "cuda"

[onnx_runtime.musvit.cuda]
arena_extend_strategy = "kNextPowerOfTwo"
use_tf32 = 0
tunable_op_enable = false
tunable_op_tuning_enable = false
```

This prevents the shared `auto` order from selecting TensorRT and FP16 options
that were not part of the model's parity evidence. An explicit CPU selection is
supported. If CUDA is explicitly selected but unavailable, the resolved session
provider list must contain only `CPUExecutionProvider`. This is an observable
result contract, not a constraint on how the shared provider builder examines
its candidates internally. CPU generation is expected to be much slower.

Before creating a real CUDA session, the shared session loader calls
`onnxruntime.preload_dlls()` when that API is available. Provider enumeration is
not proof that the CUDA runtime DLLs are loadable; without preloading, the first
inference can fail even though ONNX Runtime advertises
`CUDAExecutionProvider`.

MuSViT overrides the shared CUDA arena and TunableOp defaults. Complete-prefix
decoding changes the decoder sequence shape after every token. With
`kSameAsRequested`, repeated shape growth caused severe allocation churn and
eventual GPU allocation failure on the pinned validation page; the normal
`kNextPowerOfTwo` growth policy completed the same 1,249-token sequence without
that failure. Online per-shape kernel tuning is also disabled because generation
does not reuse a stable decoder shape. These options constrain session behavior
only; they do not change the shared provider-selection fallthrough.

Both sessions are created once for the job. Pages are processed sequentially.
There is no page batch because both graphs are fixed at batch size one and
autoregressive output lengths differ.

The provider list recorded in logs, metadata, and manifests comes from each
created session's `get_providers()` result, not merely the requested provider
list. Encoder and decoder must report the same ordered provider list or bundle
loading fails before page processing.

### Syntax-Constrained Greedy Decode

Unconstrained argmax can select a row break before all active spines have
fields, select a field separator after a reciprocal with no pitch or rest, mix
data and interpretation fields on one record, or select EOS before the Kern
document terminates. Those outputs are not convertible OMR results.

Generation therefore remains one greedy decoder call per token, but masks only
tokens that cannot extend the current prefix into a syntactically valid Kern
document:

1. decoding starts with two active root spines because the training parser
   strips the two-spine exclusive interpretation;
2. every row has one record class: interpretation, barline, or data;
3. a row may end only after it has exactly one field for every active spine;
4. an interpretation or barline field is one complete vocabulary token;
5. a data field may end only when it is `.` alone, or when every space-separated
   chord component contains a pitch token or `r`;
6. `<s>` may open another chord component only after a pitched component;
7. `<t>` is allowed only after a complete non-final field;
8. `<b>` is allowed only after a complete final field and a valid spine
   operation row;
9. spine split, join, exchange, and termination rows update the active-spine
   count before the next row;
10. EOS is allowed only after a newline-terminated row in which every active
    spine emitted `*-`.

At every step, if the normal argmax token is allowed, it remains selected
unchanged. If it is not allowed, argmax is taken over the remaining finite
logits. This is not beam search, does not add decoder calls, and does not
post-process an already generated score. If no finite legal token exists,
generation fails explicitly with the prefix state.

The constraint recognizes only syntax. It never chooses a hard-coded pitch,
rest, duration, barline, or voice. The next token always comes from the model's
own logits. Valid published greedy sequences therefore retain exact token
parity.

## Input And PDF Flow

Supported image extensions remain the project's normal score-image set,
including PNG, JPEG, BMP, TIFF, and WebP.

Input behavior:

- image: yield one page;
- PDF: yield `PdfRenderPage` values lazily;
- directory: discover supported images and PDFs in deterministic relative-path
  order, honoring `recursive`.

PDF behavior:

1. call `iter_pdf_pages_high_quality(pdf_path, dpi=pdf_dpi, image_format="PNG")`;
2. pass one rendered Pillow image into preprocessing;
3. complete recognition and exports for that page;
4. close the page image in `finally`;
5. request the next page only after the current page is released;
6. after the PDF iterator closes, aggregate the ordered page scores only when
   every page is current and successful.

No list of rendered PDF pages or preprocessed page tensors may accumulate.

PDF DPI controls rasterization before the fixed model resize. Default remains
144. Raising DPI does not change the model's 1024 input and should not be
presented as a guaranteed accuracy improvement.

## BeKern To Kern Contract

The model predicts BeKern tokens, not MusicXML or MIDI.

The Polish Scores training source starts with two `**ekern` spines. Its
training parser removes the exclusive interpretation line, splits `@` and `·`
semantic separators into BeKern tokens, and removes part of the terminal
envelope. Joining predicted semantic tokens therefore reconstructs a standard
Kern body, not a complete standalone document.

For each generated page:

1. keep the complete token-id sequence, including BOS and optional EOS;
2. decode content ids through `i2w`;
3. map `<s>` to a literal space, `<t>` to tab, and `<b>` to newline;
4. concatenate all other token strings without separators;
5. preserve the line-ending-normalized decoded body as the diagnostic
   candidate;
6. apply the exact envelope reconstruction below;
7. write the reconstructed result as `score.krn`, or write the untouched
   diagnostic candidate when envelope reconstruction fails;
8. validate spine counts and spine operations before final export.

The envelope reconstruction is normative, not pseudocode to be generalized:

```python
text = decoded_text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
if not text:
    raise KernStructureError("decoded Kern body is empty")

lines = text.split("\n")
if lines[0] == "**kern\t**kern":
    pass
elif lines[0].startswith("**"):
    raise KernStructureError("unsupported exclusive interpretation")
else:
    lines.insert(0, "**kern\t**kern")

if any(line.startswith("**") for line in lines[1:]):
    raise KernStructureError("exclusive interpretation outside the first line")

terminal_fields = lines[-1].split("\t")
if terminal_fields and all(field == "*-" for field in terminal_fields):
    pass
elif lines[-1] == "*-\t":
    lines[-1] = "*-\t*-"
else:
    raise KernStructureError("missing a complete Kern spine terminator")

kern_text = "\n".join(lines) + "\n"
```

Only an exact existing `**kern\t**kern` header is accepted. `**ekern`, any
other `**` interpretation, and a one-spine header are structural failures. The
only permitted terminal repair is the exact final line `*-\t` to `*-\t*-`.
Otherwise, the existing terminal row must contain exactly one `*-` for every
active spine. A page that ends with three active spines therefore retains the
valid `*-\t*-\t*-` row. A lone `*-` for a two-spine page, a missing terminator,
or any other partial row fails. No internal line is repaired heuristically.

An envelope error does not authorize a guessed replacement. The diagnostic
`score.krn` in that case is exactly `text + "\n"` from the code above, and
metadata marks it invalid. Only `kern_text` that completes envelope and spine
validation may enter the music21 conversion path.

The two-spine header is part of this pinned Polish Scores model profile. The
terminal width is derived from validated internal spine operations, not
inferred from visual content.

### Golden Training Fixture

Envelope tests include a checked-in fixture derived from
`PRAIG/polish-scores@b3170c8b8f322885b566efe9e264af9328b5603f`,
validation row `0`. Tests must not download the dataset. The fixture stores the
post-training-parser token body and the exact expected Kern bytes.

A representative fragment, preserving the real row's semantic token behavior,
is:

```text
*clefG2 <t> *clefG2 <b>
*k[b-e-a-] <t> *k[b-e-a-] <b>
. <t> 32 q q gg ( / L <b>
=|| <t> =|| <b>
*- <t>
```

Its expected reconstruction, shown with escaped tabs/newlines, includes:

```python
expected = (
    "**kern\t**kern\n"
    "*clefG2\t*clefG2\n"
    "*k[b-e-a-]\t*k[b-e-a-]\n"
    ".\t32qqgg(/L\n"
    "=||\t=||\n"
    "*-\t*-\n"
)
```

The implementation fixture uses the complete pinned sample body rather than
only this displayed fragment. The golden assertion compares the complete UTF-8
output exactly, including tabs and the final newline.

Allowed normalization is limited to reversing the training transform and
restoring the known document envelope. The implementation must not invent
notes, rests, durations, accidentals, measures, or internal spine operations.
If the body is structurally invalid, retain tokens, the candidate `score.krn`,
and metadata but do not produce final symbolic formats.

An EOS-free result at the token limit is `truncated`. Truncated pages retain
diagnostic files and are not exported to MusicXML or MIDI.

Before `music21.converter.parseData`, every data event is checked with the same
note-event semantics used by music21. A reciprocal-only field such as `16` is a
page failure, not a warning to suppress and not an instruction to invent
`16r`. This preflight prevents music21's permissive Humdrum parser from warning
and silently dropping an event.

## Symbolic Export Reuse

### Correct Reuse Boundary

The planned MuScriptor enriched-export design uses an audio-derived
`AnalyzedScore` containing seconds, beat mappings, estimated velocity, and
instrument events. MuSViT OMR instead produces notation with measures, rests,
voices, ties, clefs, keys, and spine operations.

Converting OMR output into MuScriptor's `AnalyzedScore` merely to call its
exporters would discard notation and make MusicXML worse. That design is
rejected.

`module.music_export` is the single owner of the now-shared mechanics:

- an `ExportJob` containing format, final target, writer callback, and validator
  callback;
- suffix-preserving sibling temporary-file creation and cleanup;
- validation before atomic replace;
- independent execution and `ExportStatus` reporting for every requested
  format;
- generic MusicXML and MIDI writers for an existing `music21.stream.Score`.

Output naming, requested-format parsing, page/job manifests, and domain metadata
remain with each caller. `module.music_export` must not import
`module.muscriptor_tool`, `module.sheet_music_omr`, `AnalyzedScore`, or model
runtime code.

This ownership change is effective now, not a deferred refactor. The
2026-07-19 enriched-export design and implementation plan are amended as part
of this spec change: MuScriptor's `musicxml_export.py` becomes only the
`AnalyzedScore -> music21.stream.Score` adapter, while common transaction,
validation, and music21 file-writing logic lives in `module.music_export`.

### OMR Score Adapter

OMR uses:

```text
validated score.krn
  -> music21 Humdrum parser
  -> music21.stream.Score
  -> shared export service
       -> MusicXML writer
       -> MIDI writer
```

MusicXML and MIDI must be written from the same parsed notation score.
MusicXML serialization uses `makeNotation=False`: the Humdrum parser already
constructed the measures and dynamic spine topology, and a second notation
pass can reinsert a measure after a valid `*^` split. This is an exporter
setting, not a rewrite of recognized notation.

Before shared export, the OMR adapter performs one notation-only
reconstruction pass over the already parsed score:

- the pinned two root spines become two `PartStaff` objects in one brace
  `StaffGroup` with joined barlines, so MusicXML contains one piano part with
  `<staves>2</staves>` rather than two unrelated instruments;
- accidental display status is recalculated per staff with key-signature and
  measure state, without changing any pitch or duration;
- explicit primary `L`/`J` beam groups are preserved, while omitted secondary
  levels in lazy Kern beaming are completed from note durations inside those
  groups; explicit higher-level beams are not replaced;
- source `K` and `k` partial-beam directions are restored as right and left
  respectively because the pinned music21 Humdrum parser maps both markers to
  a right hook;
- clef, key, and meter changes that occur inside a measure are reanchored at
  the next parsed Kern event offset; context changes immediately before a
  barline remain next-measure boundary changes;
- grace notes that music21 leaves directly on a `Part` after a boundary clef
  change are reinserted into their Kern-declared measure at the following
  principal-event offset, or at measure end when they precede a barline;
- hidden alignment rests remain hidden and use a numeric voice only when the
  source measure already contains voices, keeping joined-staff MusicXML
  unambiguous on roundtrip.

This pass must not add, remove, repitch, retime, split, or merge recognized
notes and rests. It must not call `makeNotation`. MusicXML and MIDI are still
written from the same reconstructed score, and notation-only attributes must
not change the MIDI event stream.

The common MIDI writer first uses music21's normal repeat expansion. Model
output can contain a structurally parseable but unbalanced repeat sequence; in
that case, and only when music21 raises `repeat.ExpanderException`, it retries
from a deep copy with repeat barlines, repeat expressions, and repeat brackets
removed. The retry preserves the original parsed score for MusicXML and emits
the measures once in written order. A successful fallback is recorded as a
format-specific warning in page metadata, never hidden as a normal export.

MuScriptor may continue using its direct `mido` MIDI writer for exact
audio-time semantics. OMR must not call that audio-specific writer. The shared
music21 writers and export service are reusable; MuScriptor's direct MIDI
serialization remains score-type specific.

Do not add Verovio as a second conversion path unless a focused test proves
that `music21` cannot parse a valid model-produced Kern document that Verovio
can parse. One validated path is better than silent fallback branches.

### Validation

After writing:

- MusicXML must be re-opened by `music21` and contain at least one part and one
  note or rest;
- MIDI must have a valid header and parse without error; when the parsed score
  contains notes, MIDI must contain matching note events, while a valid
  rest-only score may contain no note events;
- malformed repeat fallback must leave the source score unchanged and return a
  warning through `ExportStatus`;
- a failed format is removed from its temporary path and recorded as failed;
- success in one format does not erase a failed format's diagnostics.

## Output Contract

CLI:

```text
.\.venv\Scripts\python.exe -m module.sheet_music_musvit INPUT_PATH \
  --output_format musicxml|midi|both
```

`--output_dir` is optional and defaults to no literal path in config or the
GUI. When omitted:

- a file input resolves to `INPUT_PATH.parent / "musvit_omr_output"`;
- a directory input resolves to `INPUT_PATH / "musvit_omr_output"`.

An explicit output directory still replaces that resolved root.

Image:

```text
INPUT_PARENT/musvit_omr_output/
  score.png/
    tokens.json
    score.krn
    score.musicxml
    score.mid
    metadata.json
  manifest.json
```

PDF:

```text
INPUT_PARENT/musvit_omr_output/
  score.pdf/
    score.musicxml
    score.mid
    metadata.json
    page_0001/
      tokens.json
      score.krn
      score.musicxml
      score.mid
      metadata.json
    page_0002/
      ...
  manifest.json
```

Directory inputs preserve the source-relative parent path before the
source-name directory so equal basenames do not collide.

Only requested final formats are written. `tokens.json`, `score.krn`, and
`metadata.json` are always retained after generation begins. They are
diagnostic symbolic artifacts, not embeddings.

`tokens.json` includes:

```json
{
  "token_ids": [100, 159, 29, 159, 132, 183],
  "tokens": ["*M4/4", "<t>", "*M4/4", "<b>"],
  "terminated_by_eos": true,
  "truncated": false
}
```

`metadata.json` includes:

```json
{
  "schema_version": 2,
  "source_path": "score.pdf",
  "source_type": "pdf_page",
  "page_number": 1,
  "page_count": 2,
  "model_repo_id": "bdsqlsz/qinglong-musvit-1.0",
  "model_revision": "6f47cefe0e736fbdd0eab9e8bc8d4602e81939fe",
  "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "resolution": [1024, 1024],
  "generated_token_count": 1234,
  "terminated_by_eos": true,
  "truncated": false,
  "kern_status": "ok",
  "outputs": {
    "musicxml": "score.musicxml",
    "midi": "score.mid"
  },
  "failures": {}
}
```

For a PDF, page directories remain diagnostic and resumable units. The symbolic
files directly inside the PDF source directory are the only whole-document
result.

### PDF Score Aggregation

PDF aggregation is all-or-nothing:

1. every rendered page must have a current successful page signature and every
   requested page export;
2. pages are consumed in ascending page number;
3. every parsed page must expose the same ordered, unique part ids;
4. the validated Kern document, not music21's per-part measure-array length,
   defines the page's common measure slots: global barline rows close slots,
   while a leading barline before any data only opens the first measure;
5. spine split, exchange, and join rows track each field's original
   `spine_0`/`spine_1` ownership while slot occupancy is collected; a
   cross-part joined path may terminate but must not carry later data;
6. a slot containing a recognized event for a part must map to a non-empty
   music21 measure. A Kern null-only slot may consume music21's zero-duration
   structural measure or, when music21 omits it, create one silent transport
   measure. Unmapped parsed measures and missing event-bearing measures fail;
7. Kern interpretation rows, not the container chosen by music21's Humdrum
   parser, define clef, key, meter, and metric-symbol placement. Header
   interpretations attach to the first slot; a boundary interpretation after
   a part's last event attaches to the next slot. Data after such an
   interpretation before the barline is an unsupported mid-measure context
   change and fails explicitly;
8. parsed clef, key, and meter objects are stripped from their incidental
   music21 containers and re-anchored to the Kern slot. `*met(c)`/`*met(C)`
   preserve a common-time symbol and `*met(c|)`/`*met(C|)` preserve cut time;
9. an unchanged clef, key, or meter repeated by the next PDF page header is
   omitted from the merged score, while a changed value is preserved exactly;
10. image and individual PDF-page exports use this same one-page normalizer,
   so omitted null-only measures become hidden transport silence rather than
   visible full-measure rests;
11. aligned measures are deep-copied page by page and renumbered globally;
12. for each matching Kern slot, the longest recognized part duration is the
   alignment duration;
13. a shorter part is extended only with hidden, silent rests. A measure that
   already has voices receives a dedicated numeric padding voice; an unvoiced
   measure receives the padding directly so MusicXML does not mix tagged and
   untagged voices. Any complex padding duration is split into MusicXML-
   expressible rest components, while the count of logical gaps and their
   total duration are recorded as a document warning;
14. no recognized note, chord, visible rest, tie, context change, expression,
    or barline is semantically rewritten; only a repeated context with the same
    effective value is omitted;
15. MusicXML is serialized from the already measured aggregate without asking
   music21 to remake notation, and MIDI is written from the same aggregate
   score;
16. both aggregate formats pass the normal shared readback validators before
    document success is recorded.

Hidden alignment rests and Kern-declared null-only measures are transport
structure, not OMR content recovery. They never replace a recognized event. If
a part, an event-bearing measure, or an unambiguous spine-to-part mapping is
absent, aggregation fails instead of synthesizing music. If any page or
aggregate format fails, no current whole-document result is reported. Known
aggregate targets from an older signature are removed so stale files cannot
masquerade as the current PDF.

## Resume, Failure, And Exit Semantics

A page is complete only when:

- its metadata signature matches the current source, page, model revision,
  graph/config hashes, preprocessing contract, effective token limit, PDF DPI,
  exporter schema, and requested formats;
- all requested final files exist and pass lightweight validation.

Old embedding outputs never satisfy this signature.

A PDF document result is complete only when its metadata signature covers the
ordered page signatures, aggregate schema version, requested formats, and every
validated aggregate target. Page-level resume never by itself claims that the
PDF is complete.

The context/grace reconstruction change sets both the page exporter schema and
PDF aggregate schema to version `6`. Version `5` outputs are regenerated rather
than resumed.

Failure behavior:

- model or bundle contract failure: stop before processing pages;
- image/PDF open failure: record source failure and continue other directory
  inputs;
- before a non-resumed page attempt, remove its known token, Kern, MusicXML,
  and MIDI artifacts so an older result cannot survive a current failure;
- generation exception: record page failure and continue;
- no EOS at token limit: mark truncated and skip final export;
- Kern structural failure: keep tokens, record failure, skip final export;
- one format conversion failure: keep successful formats and diagnostics,
  record the failed format;
- any failed PDF page blocks and invalidates that PDF's aggregate outputs while
  preserving every page diagnostic;
- a PDF aggregate conversion failure records a source-level failure and removes
  current aggregate targets;
- any requested page/format failure makes the final process exit nonzero after
  writing the complete manifest.

Writes use temporary files followed by atomic replace. A failed page rerun does
not replace a previously valid page format, but document-level aggregate targets
are removed when their ordered page signature is no longer current.

## Configuration

Replace the current `[musvit]` embedding settings:

```toml
[musvit]
repo_id = "bdsqlsz/qinglong-musvit-1.0"
revision = "6f47cefe0e736fbdd0eab9e8bc8d4602e81939fe"
model_dir = "huggingface"
output_format = "musicxml"
pdf_dpi = 144
recursive = true
skip_completed = true
overwrite = false
force_download = false
```

Remove:

```text
batch_size
preprocess_mode
```

`max_tokens` is an optional advanced safety override and is absent from the
default TOML. An absent value resolves at runtime to the validated custom
`config.maxlen`; an explicit value must be in `2..config.maxlen`. The effective
resolved value remains part of resume signatures. It is not a decoding-quality
control.

CLI validation rejects non-positive PDF DPI, unsupported output formats, and
token limits outside `2..config.maxlen` before downloading or loading the model.
Pipeline-construction failures are handled as normal nonzero CLI failures.

## GUI

The Tools page keeps one "Sheet Music" entry but changes it from embedding
extraction to OMR transcription.

Visible controls:

- image/PDF/directory input;
- optional output directory, initially blank;
- output format: MusicXML, MIDI, or Both;
- PDF DPI;
- recursive directory scan;
- skip completed;
- overwrite.

Remove:

- embedding language in all translations;
- ONNX embedding repository selector;
- preprocessing selector;
- batch-size control;
- "Start Embedding" action text;
- advanced/debug embedding toggle proposed by the older spec.

The model repository and pinned revision remain configurable in TOML/CLI for
development but are not normal GUI choices. A fixed product model does not need
a user-facing repository form.

Progress should report current source/page and generated token count at a
throttled interval. It must not log every token.

Leaving the GUI output selector blank omits `--output_dir`; the backend resolves
the input-local default. The GUI must not prefill
`workspace/musvit_omr_output`.

## Dependencies

Keep routine inference inside the `musvit-onnx` profile:

```text
onnxruntime
numpy
pillow
huggingface-hub
PyMuPDF
qinglong-captions[music-export]
```

Create the neutral `music-export` extra now:

```toml
music-export = [
    "mido>=1.3.0",
    "music21==9.9.2; python_version == '3.10'",
    "music21==10.5.0; python_version >= '3.11' and python_version < '3.13'",
]
```

Both `musvit-onnx` and `muscriptor-local` reference
`qinglong-captions[music-export]`. This gives the shared package one dependency
owner without making OMR install the MuScriptor model or making MuScriptor
install ONNX Runtime. `mido` is used by the common MIDI readback validator; it
does not make MuScriptor's audio-specific MIDI writer part of the shared layer.

Do not add:

```text
torch
torchvision
transformers
lightning
wandb
datasets
cairosvg
```

## Tests

### ONNX Bundle And Runtime

- both `hf_hub_download` and every `list_repo_files` call, including external
  data discovery, receive the pinned revision;
- revision-specific cache paths do not reuse another revision's files;
- an existing target with a revision still reaches the revision-aware
  downloader instead of taking the unversioned fast path;
- default graph files match the expected size/hash manifest;
- fake sessions with wrong names, dtypes, or shapes fail before generation;
- missing/unknown preprocessor keys and the legacy `size` key fail before
  generation;
- a fixed RGB fixture produces an array exactly equal to a bilinear,
  `numpy.float32(255.0)` reference tensor; substituting bicubic must fail this
  golden assertion;
- a config fixture containing `max_length=20`, `bos_token_id=null`, and
  `eos_token_id=null` still resolves `maxlen=7512`, BOS `100`, and EOS `183`
  from the custom fields and vocabulary;
- all other config/vocab/preprocessor mismatches fail before generation;
- the encoder runs once per page;
- decoder prefixes grow from `[BOS]` to the complete sequence;
- a valid unconstrained greedy sequence is byte-identical under syntax
  constraints;
- an early row break with too few spine fields selects the model's next-highest
  legal token without another decoder call;
- a separator after reciprocal-only `16`, a mixed interpretation/data row, and
  early EOS are masked;
- constrained greedy decoding stops at EOS only after a complete all-spine
  terminator row;
- no-finite-legal-token prefixes fail explicitly;
- reaching the effective model/config token limit reports truncation;
- unknown token ids fail explicitly;
- the default runtime selects CUDA rather than TensorRT when CUDA is present;
- explicit `execution_provider="cuda"` with only
  `CPUExecutionProvider` available resolves to CPU only and never adds a
  TensorRT provider;
- bundle metadata uses the providers reported by the created sessions and
  rejects encoder/decoder provider-list disagreement;
- real CUDA session creation preloads provider DLLs when ONNX Runtime exposes
  `preload_dlls`;
- the MuSViT CUDA arena uses `kNextPowerOfTwo` for growing decoder prefixes;
- the MuSViT CUDA configuration disables both TunableOp execution and tuning.

### BeKern And Export

- special tokens reconstruct spaces, tabs, and lines exactly;
- the complete pinned Polish Scores validation-row fixture reconstructs exactly,
  including tabs and final newline;
- a headerless two-spine body receives exactly one `**kern\t**kern` header;
- an existing exact `**kern\t**kern` header is not duplicated;
- `**ekern`, other exclusive interpretations, and later exclusive
  interpretation rows fail structurally;
- only an exact final `*-\t` is expanded to `*-\t*-`;
- existing complete two- and three-spine terminators are unchanged, while
  missing or width-mismatched terminators fail;
- reciprocal-only data events fail before music21 can warn and drop them;
- malformed inner spine structure is rejected, not repaired;
- source `K` and `k` partial beams remain right and left hooks after parsing;
- truncated output never produces MusicXML/MIDI;
- a failed fresh attempt removes symbolic files left by an older page result;
- a minimal valid Kern page parses and writes both formats;
- a valid dynamic-spine page writes MusicXML without a second notation pass;
- a two-spine page writes one MusicXML piano part with two staves, a brace, and
  joined barlines;
- accidental reconstruction hides key-implied repetitions and displays a
  natural and a same-measure return to the key-signature accidental;
- lazy primary beam groups receive only their missing secondary levels;
- notation reconstruction leaves the MIDI event stream unchanged;
- MusicXML and MIDI validators reject empty or malformed outputs;
- one format failure preserves the other format and all diagnostics.

### Input And PDF

- image, PDF, and directory discovery is deterministic;
- PDF pages are requested lazily;
- each rendered page image closes after processing;
- render, inference, and export failures still close the PDF document;
- a two-page PDF produces `page_0001` and `page_0002` without materializing
  both images;
- a fully successful two-page PDF produces one validated aggregate MusicXML
  and MIDI in page order;
- a failed page leaves diagnostics but produces no current aggregate;
- Kern barline slots align a pickup and a dynamic-spine null-only measure even
  when music21 omits the corresponding leading measure in one part;
- page MusicXML writes those null-only slots as hidden silence, not visible
  full-measure rests;
- a clef emitted after a measure's final event is re-anchored to the following
  Kern slot even when music21 leaves it at Part scope or in the prior measure;
- repeated page-header clef/key/meter values are omitted, changed values remain,
  and common/cut-time metric symbols survive MusicXML roundtrip;
- an event-bearing part/measure mismatch still fails instead of synthesizing
  music;
- deterministic hidden padding aligns shorter recognized measures and is
  recorded in document metadata;
- complex hidden-padding durations are split into MusicXML-expressible rests;
- hidden padding in a joined grand staff roundtrips without missing-voice
  warnings;
- equal filenames under different source directories do not collide.

### GUI And Process Runner

- the existing script key still maps to `module.sheet_music_musvit` and the
  `musvit-onnx` extra;
- `musvit-onnx` references the neutral `music-export` extra but not
  `muscriptor-local`;
- GUI arguments contain output format and PDF settings;
- the GUI output selector starts blank and an omitted CLI output resolves beside
  or inside the input;
- GUI boolean arguments, including overwrite, always pass an explicit positive
  or negative CLI flag;
- invalid PDF DPI and token limits fail before recognizer construction;
- GUI arguments contain no batch/preprocess/embedding options;
- all supported translations describe OMR and MusicXML/MIDI output;
- docs no longer advertise sheet-music embeddings.

### Opt-In Real Smoke

An opt-in test may download the pinned public model and run one known score
page:

```text
RUN_MUSVIT_OMR_SMOKE=1 .\.venv\Scripts\python.exe -m pytest \
  tests/test_sheet_music_musvit_omr_smoke.py -q
```

It must:

- request the exact dataset revision and require the returned asset URL to identify
  that same revision
  `b3170c8b8f322885b566efe9e264af9328b5603f` and verify the raw image
  SHA-256 before inference;
- load both ONNX graphs;
- reach EOS;
- write structurally valid `score.krn`;
- produce parseable MusicXML and MIDI;
- record the exact repo revision and providers.

The large download and autoregressive runtime keep this test out of the normal
unit suite.

## Acceptance Criteria

1. The Sheet Music GUI accepts image/PDF/directory input and offers MusicXML,
   MIDI, or Both.
2. The default model is
   `bdsqlsz/qinglong-musvit-1.0@6f47cefe0e736fbdd0eab9e8bc8d4602e81939fe`.
3. Runtime uses the shared two-model ONNX base and no PyTorch training code.
4. Preprocessing strictly consumes `image_size` and bilinear interpolation and
   is tensor-identical to the pinned reference fixture.
5. Encoder inference runs exactly once per page.
6. PDF pages are rendered, inferred, exported, and released one at a time.
7. A successful PDF produces one whole-document MusicXML/MIDI result only after
   every page succeeds.
8. Successful pages contain `tokens.json`, `score.krn`, metadata, and every
   requested final format.
9. MusicXML is one joined two-staff piano part; accidental and lazy-beam
   reconstruction changes no recognized note semantics or MIDI events.
10. MusicXML and MIDI are validated before being reported as successful.
11. Truncated or invalid Kern output never masquerades as a successful score.
12. Resume signatures cannot confuse old embeddings with OMR results.
13. The normal tool writes no `embedding.npz` and exposes no embedding mode.
14. Shared export orchestration is neutral; OMR does not depend on the
    MuScriptor audio model or `AnalyzedScore`.
15. Focused unit/integration tests pass in the project `.venv`; the real model
    smoke passes when explicitly enabled.

## Implementation Order

1. Replace old embedding contract tests with the new CLI/config/output contract.
2. Add revision propagation and revision-specific caching to the shared ONNX
   artifact layer.
3. Add the strict preprocessor loader and bilinear reference-tensor tests.
4. Add the two-graph model contract and fake-session constrained-greedy decoder
   tests.
5. Add BeKern reconstruction and strict Kern envelope/structure tests.
6. Implement the shared `module.music_export` prerequisite once and treat
   Prerequisite Task 0 in the revised MuScriptor plan as satisfied by the same
   package and contract tests; do not create a second service.
7. Add MusicXML/MIDI score writers and validators.
8. Add the page pipeline, streamed PDF lifecycle, and all-or-nothing aggregate.
9. Add the grand-staff, accidental-display, lazy-beam, and padding-voice
   reconstruction pass and bump both export resume schemas.
10. Replace the existing sheet-music CLI implementation in place.
11. Replace GUI controls, translations, docs, and config defaults.
12. Run focused tests, then the opt-in real model smoke in the local `.venv`.

## Risks

### Full-Prefix Decode Cost

The decoder recomputes the entire prefix for every generated token. Total
decoder work grows steeply with sequence length. The published ten-page run
averaged about 21.6 seconds per page on CUDA/CPU provider configuration, and CPU
can be substantially slower. Syntax constraints do not add decoder calls and
preserve parity whenever the unconstrained token is legal. This cost is accepted
for version one.

### Model Scope

The evidence covers ten Polish Scores validation pages. Accuracy on degraded
scans, handwriting, dense orchestral scores, unusual notation, or non-piano
layouts is unknown.

### Invalid Symbolic Sequences

Autoregressive output can be locally plausible but structurally invalid.
Syntax-constrained greedy decoding prevents locally impossible separators and
envelopes without inventing content. Strict Kern parsing and per-format
validation remain required because token metrics and local grammar alone do not
guarantee convertible notation.

### Missing Bundle Metadata

The Hub upload omits the reference exporter's signed `metadata.json`. Pinning
the revision, verifying known graph hashes, and validating live graph/config
contracts are required to avoid mixed artifacts.

### License Metadata

The fine-tuned model card does not declare a license tag. The base MuSViT
repository states CC BY-NC-SA 4.0. Distribution and commercial-use claims must
remain conservative until the fine-tuned repository declares its applicable
license explicitly.

### Shared Export Work Is Not Yet Implemented

The project has a design and implementation plan for enhanced MuScriptor
MusicXML/MIDI exports, but the neutral shared export modules do not yet exist on
the reviewed branch. OMR implementation must establish that boundary rather
than importing future audio-specific modules by name.

## References

- Fine-tuned ONNX model:
  `https://huggingface.co/bdsqlsz/qinglong-musvit-1.0`
- Reference branch:
  `https://github.com/sdbds/MuSViT/tree/codex/full-page-omr-throughput`
- Reference ONNX design:
  `https://github.com/sdbds/MuSViT/blob/codex/full-page-omr-throughput/docs/superpowers/specs/2026-07-18-full-page-omr-onnx-export-design.md`
- Polish Scores dataset:
  `https://huggingface.co/datasets/PRAIG/polish-scores`
- Previous embedding design:
  `docs/superpowers/specs/2026-07-02-musvit-onnx-sheet-music-tool-design.md`
- Shared PDF rendering design:
  `docs/superpowers/specs/2026-07-02-streaming-pdf-rendering-design.md`
- Amended enriched music export design:
  `docs/superpowers/specs/2026-07-19-music-analysis-enriched-midi-musicxml-design.md`
- Amended enriched music export implementation plan:
  `docs/superpowers/plans/2026-07-19-enriched-music-exports-implementation.md`
