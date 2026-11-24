"""LiquidAI LFM VL local ONNX provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from module.onnx_runtime.artifacts import build_component_filename, build_local_model_dir, download_onnx_artifact_set
from module.onnx_runtime.config import resolve_tool_runtime_config
from module.onnx_runtime.session import load_session_bundle
from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.local_vlm import lfm3_vl_onnx
from module.providers.local_vlm_base import LocalVLMProvider
from module.providers.registry import register_provider

_ONNX_TYPE_TO_DTYPE = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}


def _replace_present_name(name: str) -> str:
    return (
        name.replace("present_conv", "past_conv")
        .replace("present.", "past_key_values.")
        .replace("present_key_values", "past_key_values")
    )


@register_provider("lfm_vl_local")
class LFMVLLocalProvider(LocalVLMProvider):
    """LiquidAI LFM ONNX-backed local VLM provider."""

    default_model_id = "LiquidAI/LFM3-VL-2B-ONNX"

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        return getattr(args, "vlm_image_model", "") == "lfm_vl_local" and mime.startswith("image")

    def _component_variant(self, component: str) -> str:
        explicit = str(self.model_config.get(f"{component}_variant", "") or "").strip()
        if explicit:
            return "fp16" if explicit.lower() == "auto" else explicit

        if component in {"embed_tokens", "embed_images"}:
            shared = str(self.model_config.get("encoder_variant", "") or "").strip()
            if component == "embed_tokens" and shared.lower() in {"q4", "q8"}:
                return "fp16"
            return shared

        return str(self.model_config.get("decoder_variant", "") or "").strip()

    def _legacy_component_files(self) -> dict[str, str]:
        return {
            "embed_tokens": build_component_filename("embed_tokens", self._component_variant("embed_tokens")),
            "embed_images": build_component_filename("embed_images", self._component_variant("embed_images")),
            "decoder": build_component_filename("decoder", self._component_variant("decoder")),
        }

    def _load_model(self):
        from huggingface_hub import list_repo_files

        model_id = self.model_id
        runtime_config = resolve_tool_runtime_config(
            self.ctx.config,
            tool_name=self.name,
            legacy=self.model_config,
        )
        local_dir = runtime_config.resolve_model_cache_dir(model_id)
        repo_files = list_repo_files(model_id)
        if any(path.startswith("onnx/vision_encoder") and path.endswith(".onnx") for path in repo_files):
            runtime_config, component_files = lfm3_vl_onnx.resolve_bundle(
                runtime_config, str(self.model_config.get("decoder_variant") or "auto"),
            )
            if runtime_config.model_cache_dir:
                # A legacy cache in this directory may contain a different decoder with the same filename.
                local_dir = build_local_model_dir(runtime_config.model_cache_dir, model_id)
            preprocessing = lfm3_vl_onnx.load_preprocessing(
                model_id, local_dir, force_download=runtime_config.force_download, logger=self.ctx.console.print,
            )
        else:
            from transformers import AutoProcessor

            from utils.transformer_loader import load_pretrained_component

            component_files = self._legacy_component_files()
            processor = load_pretrained_component(
                AutoProcessor, model_id, console=self.ctx.console, component_name="processor", trust_remote_code=True,
            )
            preprocessing = {"processor": processor}

        self.log(
            "Selected LFM ONNX artifacts: "
            + ", ".join(f"{name}={path}" for name, path in component_files.items()),
            "blue",
        )

        session_paths = download_onnx_artifact_set(
            model_id,
            component_files,
            repo_files=repo_files,
            local_dir=local_dir,
            force_download=runtime_config.force_download,
            logger=self.ctx.console.print,
        )
        bundle = load_session_bundle(
            bundle_key=self._model_key,
            session_paths=session_paths,
            runtime_config=runtime_config,
        )

        return {
            **preprocessing,
            "sessions": bundle.sessions,
            "session_bundle": bundle,
            "runtime_config": runtime_config,
            "local_dir": local_dir,
        }

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        if self.get_runtime_backend().is_openai:
            return self.attempt_via_openai_backend(media, prompts)

        cached = self._get_or_load_model()
        processor = cached["processor"]
        sessions = cached["sessions"]

        images = self._load_images(media)
        messages = self._build_messages(prompts, image_count=len(images))
        if "vision_encoder" in sessions:
            tokenizer = cached["tokenizer"]
            prefix = lfm3_vl_onnx.prepare_prefill(
                processor, tokenizer, sessions["vision_encoder"], messages, images, self.model_config,
            )
        else:
            tokenizer = processor.tokenizer
            prefix = {"inputs_embeds": self._legacy_prefix_embeddings(processor, sessions, messages, images)}

        tokens = self._decode_tokens(sessions=sessions, prefix=prefix, eos_token_id=getattr(tokenizer, "eos_token_id", None))
        response_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return CaptionResult(raw=response_text, metadata={"provider": self.name, "model_id": self.model_id})

    def _legacy_prefix_embeddings(self, processor, sessions, messages, images) -> np.ndarray:
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self._prepare_processor_inputs(processor, images, prompt)

        embed_images = sessions["embed_images"]
        embed_tokens = sessions["embed_tokens"]

        image_embeds = np.asarray(
            embed_images.run(
                None,
                self._filter_session_inputs(
                    embed_images,
                    {
                        "pixel_values": inputs["pixel_values"],
                        "pixel_attention_mask": inputs.get("pixel_attention_mask"),
                        "spatial_shapes": inputs.get("spatial_shapes"),
                    },
                ),
            )[0]
        )
        token_embeds = np.asarray(
            embed_tokens.run(
                None,
                self._filter_session_inputs(embed_tokens, {"input_ids": inputs["input_ids"]}),
            )[0]
        )

        return self._merge_image_embeddings(
            token_embeds=token_embeds,
            input_ids=inputs["input_ids"],
            image_embeds=image_embeds,
            image_token_id=processor.tokenizer.convert_tokens_to_ids("<image>"),
        )

    def _build_messages(self, prompts: PromptContext, *, image_count: int) -> list[dict[str, Any]]:
        user_content = [self.build_image_part() for _ in range(image_count)]
        user_content.append(self.build_text_part(prompts.user))
        messages = [self.build_message("user", user_content)]
        if prompts.system:
            messages.insert(0, self.build_message("system", [self.build_text_part(prompts.system)]))
        return messages

    def _load_images(self, media: MediaContext) -> list[Image.Image]:
        uris = [media.uri]
        pair_uri = media.extras.get("pair_uri")
        if pair_uri:
            uris.append(pair_uri)

        images: list[Image.Image] = []
        for uri in uris:
            with Image.open(uri) as image:
                images.append(image.convert("RGB").copy())
        return images

    def _prepare_processor_inputs(self, processor: Any, images: list[Image.Image], prompt: str) -> dict[str, Any]:
        processor_kwargs = {
            "images": images,
            "text": prompt,
            "return_tensors": "np",
        }
        for key in ("min_image_tokens", "max_image_tokens", "do_image_splitting"):
            value = self.model_config.get(key)
            if value not in (None, ""):
                processor_kwargs[key] = value

        try:
            raw_inputs = processor(**processor_kwargs)
        except Exception:
            processor_kwargs["return_tensors"] = "pt"
            raw_inputs = processor(**processor_kwargs)
        return {name: self._to_numpy(value) for name, value in raw_inputs.items()}

    def _to_numpy(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        if isinstance(value, (list, tuple)):
            return np.asarray(value)
        return value

    def _dtype_for_meta(self, meta: Any):
        return _ONNX_TYPE_TO_DTYPE.get(getattr(meta, "type", ""), np.float32)

    def _coerce_for_meta(self, value: Any, meta: Any) -> np.ndarray:
        array = np.asarray(value)
        return np.ascontiguousarray(array.astype(self._dtype_for_meta(meta), copy=False))

    def _filter_session_inputs(self, session: Any, values: dict[str, Any]) -> dict[str, np.ndarray]:
        feed: dict[str, np.ndarray] = {}
        for meta in session.get_inputs():
            if meta.name not in values or values[meta.name] is None:
                continue
            feed[meta.name] = self._coerce_for_meta(values[meta.name], meta)
        return feed

    def _merge_image_embeddings(
        self,
        *,
        token_embeds: np.ndarray,
        input_ids: np.ndarray,
        image_embeds: np.ndarray,
        image_token_id: int,
    ) -> np.ndarray:
        merged = np.array(token_embeds, copy=True)
        image_rows = np.asarray(image_embeds)
        if image_rows.ndim == 3 and image_rows.shape[0] == 1:
            image_rows = image_rows[0]
        if image_rows.ndim == 1:
            image_rows = image_rows.reshape(1, -1)

        image_positions = np.where(np.asarray(input_ids)[0] == image_token_id)[0]
        for index, position in enumerate(image_positions[: len(image_rows)]):
            merged[0, position] = image_rows[index]
        return merged

    def _initialize_decoder_cache(self, decoder: Any) -> dict[str, np.ndarray]:
        cache: dict[str, np.ndarray] = {}
        for meta in decoder.get_inputs():
            if not meta.name.startswith("past"):
                continue

            shape = []
            for dim in getattr(meta, "shape", []) or []:
                if isinstance(dim, int):
                    shape.append(dim)
                elif isinstance(dim, str) and "sequence" in dim.lower():
                    shape.append(0)
                else:
                    shape.append(1)

            cache[meta.name] = np.zeros(shape, dtype=self._dtype_for_meta(meta))
        return cache

    def _decode_tokens(
        self,
        *,
        sessions: dict[str, Any],
        prefix: dict[str, np.ndarray],
        eos_token_id: int | None,
    ) -> list[int]:
        decoder = sessions["decoder"]
        cache = self._initialize_decoder_cache(decoder)
        prefix_length = prefix["input_ids" if "input_ids" in prefix else "inputs_embeds"].shape[1]
        max_new_tokens = int(self.model_config.get("max_new_tokens", self.model_config.get("out_seq_length", 256)))
        generated_tokens: list[int] = []

        for step in range(max_new_tokens):
            if step == 0:
                current = prefix
            else:
                last_token = np.asarray([[generated_tokens[-1]]], dtype=np.int64)
                if "input_ids" in prefix:
                    current = {**prefix, "input_ids": last_token, "image_features": prefix["image_features"][:0]}
                else:
                    embed_tokens = sessions["embed_tokens"]
                    feed = self._filter_session_inputs(embed_tokens, {"input_ids": last_token})
                    current = {"inputs_embeds": embed_tokens.run(None, feed)[0]}

            total_length = prefix_length + step
            feed_values: dict[str, Any] = {
                **current,
                "attention_mask": np.ones((1, total_length), dtype=np.int64),
                "position_ids": np.arange(0 if step == 0 else total_length - 1, total_length, dtype=np.int64)[None, :],
                **cache,
            }

            outputs = decoder.run(None, self._filter_session_inputs(decoder, feed_values))
            logits = np.asarray(outputs[0])
            next_token = int(np.argmax(logits[0, -1]))
            generated_tokens.append(next_token)
            self._update_decoder_cache(decoder, cache, outputs)

            if eos_token_id is not None and next_token == int(eos_token_id):
                break

        return generated_tokens

    def _update_decoder_cache(self, decoder: Any, cache: dict[str, np.ndarray], outputs: list[Any]) -> None:
        for meta, value in zip(decoder.get_outputs()[1:], outputs[1:]):
            cache_name = _replace_present_name(meta.name)
            if cache_name not in cache:
                continue
            cache[cache_name] = self._coerce_for_meta(value, meta)
