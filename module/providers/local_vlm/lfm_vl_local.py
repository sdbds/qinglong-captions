"""LiquidAI LFM VL local ONNX provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from onnx_runtime.artifacts import build_component_filename, download_onnx_artifact_set
from onnx_runtime.config import resolve_tool_runtime_config
from onnx_runtime.session import load_session_bundle
from module.providers.base import CaptionResult, MediaContext, PromptContext
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

    default_model_id = "LiquidAI/LFM2.5-VL-1.6B-ONNX"

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        return getattr(args, "vlm_image_model", "") == "lfm_vl_local" and mime.startswith("image")

    def _component_variant(self, component: str) -> str:
        explicit = str(self.model_config.get(f"{component}_variant", "") or "").strip()
        if explicit:
            return explicit

        if component in {"embed_tokens", "embed_images"}:
            shared = str(self.model_config.get("encoder_variant", "") or "").strip()
            if component == "embed_tokens" and shared.lower() in {"q4", "q8"}:
                return "fp16"
            return shared

        return str(self.model_config.get("decoder_variant", "") or "").strip()

    def _component_files(self) -> dict[str, str]:
        return {
            "embed_tokens": build_component_filename("embed_tokens", self._component_variant("embed_tokens")),
            "embed_images": build_component_filename("embed_images", self._component_variant("embed_images")),
            "decoder": build_component_filename("decoder", self._component_variant("decoder")),
        }

    def _load_model(self):
        from transformers import AutoProcessor

        model_id = self.model_id
        runtime_config = resolve_tool_runtime_config(
            self.ctx.config,
            tool_name=self.name,
            legacy=self.model_config,
        )
        local_dir = runtime_config.resolve_model_cache_dir(model_id)
        component_files = self._component_files()

        self.log(
            "Selected LFM ONNX artifacts: "
            + ", ".join(f"{name}={path}" for name, path in component_files.items()),
            "blue",
        )

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        session_paths = download_onnx_artifact_set(
            model_id,
            component_files,
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
            "processor": processor,
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
        prompt = processor.apply_chat_template(
            self._build_messages(prompts, image_count=len(images)),
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self._prepare_processor_inputs(processor, images, prompt)

        embed_images = sessions["embed_images"]
        embed_tokens = sessions["embed_tokens"]
        decoder = sessions["decoder"]

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

        merged_embeds = self._merge_image_embeddings(
            token_embeds=token_embeds,
            input_ids=inputs["input_ids"],
            image_embeds=image_embeds,
            image_token_id=processor.tokenizer.convert_tokens_to_ids("<image>"),
        )
        generated_tokens = self._decode_tokens(
            decoder=decoder,
            embed_tokens=embed_tokens,
            prefix_embeds=merged_embeds,
            eos_token_id=getattr(processor.tokenizer, "eos_token_id", None),
        )

        response_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return CaptionResult(raw=response_text, metadata={"provider": self.name, "model_id": self.model_id})

    def _build_messages(self, prompts: PromptContext, *, image_count: int) -> list[dict[str, Any]]:
        user_content = [{"type": "image"} for _ in range(image_count)]
        user_content.append({"type": "text", "text": prompts.user})
        messages = [{"role": "user", "content": user_content}]
        if prompts.system:
            messages.insert(0, {"role": "system", "content": prompts.system})
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
            if meta.name in {"inputs_embeds", "attention_mask", "position_ids"}:
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
        decoder: Any,
        embed_tokens: Any,
        prefix_embeds: np.ndarray,
        eos_token_id: int | None,
    ) -> list[int]:
        cache = self._initialize_decoder_cache(decoder)
        prefix_embeds = np.ascontiguousarray(np.asarray(prefix_embeds, dtype=np.float32))
        prefix_length = int(prefix_embeds.shape[1])
        max_new_tokens = int(self.model_config.get("max_new_tokens", self.model_config.get("out_seq_length", 256)))
        generated_tokens: list[int] = []

        for step in range(max_new_tokens):
            if step == 0:
                current_embeds = prefix_embeds
            else:
                last_token = np.asarray([[generated_tokens[-1]]], dtype=np.int64)
                current_embeds = np.asarray(
                    embed_tokens.run(
                        None,
                        self._filter_session_inputs(embed_tokens, {"input_ids": last_token}),
                    )[0],
                    dtype=np.float32,
                )

            total_length = prefix_length + len(generated_tokens)
            feed_values: dict[str, Any] = {
                "inputs_embeds": current_embeds,
                "attention_mask": np.ones((1, total_length), dtype=np.int64),
                **cache,
            }
            if any(meta.name == "position_ids" for meta in decoder.get_inputs()):
                start = total_length - current_embeds.shape[1]
                feed_values["position_ids"] = np.arange(start, total_length, dtype=np.int64)[None, :]

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
