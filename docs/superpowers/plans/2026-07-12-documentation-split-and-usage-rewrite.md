# Documentation Split And Usage Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the bilingual project documentation into focused files and rewrite the usage guidance so every command matches the current repository entrypoints.

**Architecture:** `README.md` becomes the Chinese-first entrypoint for installation, quick start, workflows, safety, and troubleshooting. `README.en.md` mirrors the same operational surface in English. `CHANGELOG.md` owns release history, while GUI and configuration details remain in focused subordinate documents linked from both entrypoints.

**Tech Stack:** Markdown, PowerShell, Python module CLIs, `uv`, pytest, Ruff.

## Global Constraints

- Preserve the existing script names and CLI flags; documentation must follow the repository rather than inventing aliases.
- Python support remains `>=3.10,<3.13` as declared in `pyproject.toml`.
- GUI startup through `start_gui.ps1` uses the PEP 723 runtime in `gui/launch.py`; do not claim it reuses `.venv` or `venv`.
- Linux installation must reference the literal repository filename `0.install pwsh.sh` and quote the space.
- Never include real credentials, tokens, or local absolute paths in documentation.
- Explicitly warn that `config/env_vars.json` is plaintext local state and that `--cloud` binds the unauthenticated GUI to `0.0.0.0`.
- PowerShell wrappers must use `LOCALAPPDATA` on Windows and `HOME` or a repository-local fallback on Linux for `UV_CACHE_DIR`.

---

### Task 1: Move release history out of the entrypoint

**Files:**
- Create: `CHANGELOG.md`
- Modify: `README.md`

**Interfaces:**
- `README.md` links to `CHANGELOG.md` for release history.
- `CHANGELOG.md` retains the current 4.5 through 3.0 history without operational instructions.

- [x] **Step 1: Copy the existing release headings and entries into `CHANGELOG.md`**
- [x] **Step 2: Replace the long changelog block in `README.md` with a short release-history link**
- [x] **Step 3: Check that no version heading was lost**

Run: `rg -n '^###? ' README.md CHANGELOG.md`

Expected: operational README headings remain, release headings are present in `CHANGELOG.md`, and README no longer contains the duplicated historical block.

### Task 2: Rewrite the Chinese entrypoint

**Files:**
- Modify: `README.md`
- Create: `docs/configuration.md`
- Create: `docs/troubleshooting.md`

**Interfaces:**
- README links to `gui/README.md`, `docs/configuration.md`, `docs/openai_compatible.md`, and `docs/troubleshooting.md`.
- Usage examples use the actual `start_gui.ps1`, `0.install pwsh.sh`, `1.install-uv-qinglong.ps1`, and module CLI names.

- [x] **Step 1: Add a concise project overview and requirements**
- [x] **Step 2: Add a copy-paste quick start for Windows and Linux**
- [x] **Step 3: Rewrite usage by user goal: GUI captioning, OCR/local routes, translation, audio separation, Image2PSD, and batch scripts**
- [x] **Step 4: Document input/output expectations and the Lance dataset flow**
- [x] **Step 5: Add provider configuration, secret handling, cloud-mode warning, and dependency-profile guidance**
- [x] **Step 6: Move detailed environment/configuration and troubleshooting tables into the two focused docs**
- [x] **Step 7: Add links to the changelog, GUI manual, OpenAI-compatible provider guide, citations, and license**

### Task 3: Add an English operational mirror

**Files:**
- Create: `README.en.md`

**Interfaces:**
- The English file mirrors the Chinese README's operational sections and links to the same subordinate documents.
- It does not duplicate the release history; it links to `CHANGELOG.md`.

- [x] **Step 1: Write English requirements and quick start commands**
- [x] **Step 2: Mirror the workflow and script tables**
- [x] **Step 3: Mirror safety, configuration, and troubleshooting warnings**
- [x] **Step 4: Add a language-navigation link back to `README.md`**

### Task 4: Synchronize GUI documentation

**Files:**
- Modify: `gui/README.md`
- Modify: `gui/PARAMETERS.md`

**Interfaces:**
- GUI docs describe `uv run gui/launch.py` isolation accurately.
- Stale `step2_cache`/`step3_train`/`step4_generate` references are removed or explicitly identified as historical.

- [x] **Step 1: Correct startup commands, defaults, and cloud-mode warning**
- [x] **Step 2: Update page names and the GUI-to-script mapping**
- [x] **Step 3: Replace stale parameter mapping claims with links to current TOML/config sources and page-specific settings**
- [x] **Step 4: Add a concise GUI troubleshooting section**

### Task 5: Verify documentation against the repository

**Files:**
- Verify: all Markdown files changed above
- Verify: all PowerShell wrappers and `tests/test_install_uv_qinglong.py`

- [x] **Step 1: Run every documented `--help` command for the Python entrypoints**
- [x] **Step 2: Check Markdown links and referenced filenames**
- [x] **Step 3: Search for stale commands, wrong filenames, duplicated changelog headings, and credential-like literals**
- [x] **Step 4: Run the repository's non-optional test suite**

Run:

```powershell
rg -n '0、install|step[0-9]_(cache|tagging|train|generate)|reuses `.venv`|uv\.lock' README.md README.en.md gui docs -g '*.md'
uv pip install --python .\.venv\Scripts\python.exe --group test
python -m pytest tests -q --strict-markers -m "not optional_runtime and not gpu and not network"
```

Expected: the stale-pattern search returns no operational-doc matches, and the test suite remains green.
