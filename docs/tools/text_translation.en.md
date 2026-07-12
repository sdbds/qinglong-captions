# Text And Document Translation

Translation normalizes documents to Markdown, records chunk boundaries, and translates with a local model or OpenAI-compatible backend. Language-suffixed output files preserve the source.

```powershell
.\5.translate.ps1
python -m module.texttranslate --help
```

The profile is `translate`. The workflow imports source text, normalizes it, stores `chunk_offsets`, translates, and exports `*_lang.md`. Use `--normalize_only` first when diagnosing parsing, dependency, or output-permission problems.
