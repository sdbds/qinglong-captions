# OCR Format Helpers

These are pinned, offline copies of upstream parsing code, not model runtimes.
No model code or weights are downloaded at import time.

- `unlimited_v2.py`: Baidu Unlimited-OCR commit
  `0cdd7877868bd0e17715d742911b9ce937c53d3f`,
  `utils/process_unlimitedOCR2.py`, MIT (see `UNLIMITED_LICENSE`).
  The CLI and its imports are omitted; boundary whitespace is retained when
  collecting fragments so cross-page joins do not lose explicit spaces.
- `logics_v3.py`: Alibaba Logics-Parsing commit
  `d75d42dbd92c518a4def19ac5895792828138d35`, `inference_v3.py`,
  Apache-2.0 (see `LOGICS_LICENSE`). Only prompts, compact state, DSL parsing,
  continuation merging, Markdown/hierarchy rendering and visual crops are kept.
  Boundary whitespace is retained in `clean_content` for cross-page joins.
  Continuation merges also retain the adapter's caption `refs` lists.

Provider integration and JSON schemas live outside this directory. Upstream
benchmark-specific heading normalization is retained for Unlimited output.
