# Text And Document Translation

Translation normalizes documents to Markdown, records chunk boundaries, and translates with a local model or OpenAI-compatible backend. Language-suffixed output files preserve the source.

```powershell
.\5.translate.ps1
python -m module.texttranslate --help
```

The profile is `translate`. The workflow imports source text, normalizes it, stores `chunk_offsets`, translates, and exports `*_lang.md`. Use `--normalize_only` first when diagnosing parsing, dependency, or output-permission problems.

## Index-Translate

The GUI model selector and `$model_id` in `5.translate.ps1` support:

- `IndexTeam/Index-Translate-9B`
- `IndexTeam/Index-Translate-2B`

Hy-MT2-7B remains the default. Both Index models share an adapter with the official Chinese constrained-translation prompt, disabled thinking, Markdown and placeholder preservation, glossary input, and previous-chunk context. Automatic source detection omits the source language from the prompt; `zh_cn` and `zh_tw` select Simplified and Traditional Chinese.

Direct inference uses the Qwen3.5 model class and Transformers 5.6+ from the existing `translate` profile. It does not require vLLM. The first run downloads the selected model weights:

```powershell
python -m module.texttranslate ./datasets --model_id IndexTeam/Index-Translate-2B --target_lang zh_cn
```

To use an already running OpenAI-compatible server:

```powershell
python -m module.texttranslate ./datasets --model_id IndexTeam/Index-Translate-9B --runtime_backend openai --openai_base_url http://127.0.0.1:8000/v1 --openai_model_name served-index --target_lang zh_cn
```

Keep the official ID in `--model_id` to select the Index adapter. Set a custom server alias with `--openai_model_name`; when omitted, it defaults to `--model_id`. The server must support `chat_template_kwargs.enable_thinking=false`.

Decoding is greedy by default and retains the project's `max_new_tokens=4096` per chunk. Unfinished thinking blocks and empty translations raise an error instead of being saved. For longer input, reduce `--max_chars` or increase `--max_new_tokens` within the server's context and memory limits.

Official model and serving guides: [9B](https://huggingface.co/IndexTeam/Index-Translate-9B), [2B](https://huggingface.co/IndexTeam/Index-Translate-2B).
