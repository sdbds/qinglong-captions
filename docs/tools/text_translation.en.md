# Text And Document Translation

Translation normalizes documents to Markdown, records chunk boundaries, and translates with a local model or OpenAI-compatible backend. Language-suffixed output files preserve the source.

```powershell
.\5.translate.ps1
python -m module.texttranslate --help
```

The profile is `translate`. The workflow imports source text, normalizes it, stores `chunk_offsets`, translates, and exports `*_lang.md`. Use `--normalize_only` first when diagnosing parsing, dependency, or output-permission problems.

## Tencent Hy-MT3

The default is `tencent/Hy-MT3-7B`. Choose `tencent/Hy-MT3-1.8B` for lower memory use. Both sizes have `-FP8` variants for hardware that supports FP8 inference. The GUI and the script's `$model_id` retain the existing Hy-MT2 and Index-Translate choices. Direct inference downloads only the selected model and uses the existing `translate` profile without a model server.

```powershell
python -m module.texttranslate ./datasets --model_id tencent/Hy-MT3-1.8B --target_lang zh_cn
```

For target languages not listed in the GUI, pass the full English name on the CLI, for example `--target_lang Finnish`. Leave `--source_lang auto` to let the model infer the source language.

Local decoding is greedy by default. Pass `--temperature 0.7` to use [Tencent's recommended sampling settings](https://huggingface.co/tencent/Hy-MT3-7B#inference-and-deployment). The default output budget is 4096 tokens per chunk; include the prompt, glossary, and previous-chunk context when budgeting for Tencent's recommended 8192-token context. To use an existing OpenAI-compatible server, add `--runtime_backend openai --openai_base_url http://127.0.0.1:8000/v1`; use `--openai_model_name` when the server uses an alias.

## Index-Translate

The GUI model selector and `$model_id` in `5.translate.ps1` support:

- `IndexTeam/Index-Translate-9B`
- `IndexTeam/Index-Translate-2B`
- [`IndexTeam/Index-Nailong-9B` (NativeLong v1.1, long-document translation)](https://huggingface.co/IndexTeam/Index-Nailong-9B)

All three use the Index constrained-translation prompt with thinking disabled. They retain Markdown, placeholders, glossary input, and previous-chunk context. Automatic source detection omits the source language from the prompt; `zh_cn` and `zh_tw` select Simplified and Traditional Chinese.

Direct inference uses the existing `translate` profile without vLLM. Nailong uses the text-only Qwen3.5 model class. The first run downloads only the selected model's weights:

```powershell
python -m module.texttranslate ./datasets --model_id IndexTeam/Index-Nailong-9B --source_lang en --target_lang zh_cn
```

To use an already running OpenAI-compatible server:

```powershell
python -m module.texttranslate ./datasets --model_id IndexTeam/Index-Translate-9B --runtime_backend openai --openai_base_url http://127.0.0.1:8000/v1 --openai_model_name served-index --target_lang zh_cn
```

Keep the official ID in `--model_id` to select the Index adapter. Set a custom server alias with `--openai_model_name`; when omitted, it defaults to `--model_id`. The server must support `chat_template_kwargs.enable_thinking=false`. Nailong also needs `stop_token_ids`, as supported by vLLM.

Decoding is greedy by default and retains the project's `max_new_tokens=4096` per chunk. Unfinished thinking blocks and empty translations raise an error instead of being saved. For longer input, reduce `--max_chars` or increase `--max_new_tokens` within the server's context and memory limits.

Selecting Nailong does not switch to whole-document translation. The default chunk size is 2,200 **characters**. For more cross-paragraph context, increase `--max_chars` before normalization and budget `--max_new_tokens` for the longer translation. Reusing a normalized version keeps its stored chunk boundaries. The 262,144-**token** window is shared by the prompt, source, and output. BF16 weights alone use about 17.9 GB; runtime memory is additional. A length-truncated generation is not saved as a completed translation. See the [official inference guide](https://huggingface.co/IndexTeam/Index-Nailong-9B/blob/main/INFERENCE.md).

Official model and serving guides: [9B](https://huggingface.co/IndexTeam/Index-Translate-9B), [2B](https://huggingface.co/IndexTeam/Index-Translate-2B).
