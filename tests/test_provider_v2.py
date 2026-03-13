# -*- coding: utf-8 -*-
"""
Provider V2 单元测试

覆盖范围:
- CaptionResult 数据类
- MediaContext 数据类
- PromptContext 数据类
- RetryConfig 数据类
- ProviderCapabilities 数据类
- ProviderRegistry 单例 + 自动发现 + find_provider
- PromptResolver 全链路
- with_retry_impl 重试逻辑
- build_vision_messages 消息构建
- OCRProvider.can_handle 基类逻辑
- CloudVLMProvider 基类逻辑
- 所有 20+ 个 Provider 的 can_handle 路由
- api_handler_v2 入口函数
- kimi_code User-Agent header
"""

import inspect
import io
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# 确保 module 路径可导入
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


# ──────────────────────────────────────────────
#  CaptionResult
# ──────────────────────────────────────────────

class TestCaptionResult:

    def test_plain_text(self):
        from providers.base import CaptionResult
        r = CaptionResult(raw="hello world")
        assert r.raw == "hello world"
        assert r.description == "hello world"
        assert not r.is_structured
        assert r.get("key") is None
        assert r.get("key", 42) == 42

    def test_structured_long_description(self):
        from providers.base import CaptionResult
        parsed = {"long_description": "long", "short_description": "short"}
        r = CaptionResult(raw=json.dumps(parsed), parsed=parsed)
        assert r.is_structured
        assert r.description == "long"
        assert r.get("short_description") == "short"

    def test_structured_description_fallback(self):
        from providers.base import CaptionResult
        r = CaptionResult(raw="raw", parsed={"description": "desc"})
        assert r.description == "desc"

    def test_structured_short_description_fallback(self):
        from providers.base import CaptionResult
        r = CaptionResult(raw="raw", parsed={"short_description": "short"})
        assert r.description == "short"

    def test_structured_raw_fallback(self):
        from providers.base import CaptionResult
        r = CaptionResult(raw="raw_fallback", parsed={"other_key": "val"})
        assert r.description == "raw_fallback"

    def test_empty_raw(self):
        from providers.base import CaptionResult
        r = CaptionResult(raw="")
        assert r.description == ""
        assert not r.is_structured

    def test_metadata(self):
        from providers.base import CaptionResult
        r = CaptionResult(raw="x", metadata={"provider": "test"})
        assert r.metadata["provider"] == "test"

    def test_default_metadata(self):
        from providers.base import CaptionResult
        r = CaptionResult(raw="x")
        assert r.metadata == {}


# ──────────────────────────────────────────────
#  MediaContext
# ──────────────────────────────────────────────

class TestMediaContext:

    def test_immutable(self):
        from providers.base import MediaContext, MediaModality
        m = MediaContext(uri="/a.jpg", mime="image/jpeg", sha256hash="abc", modality=MediaModality.IMAGE)
        with pytest.raises(AttributeError):
            m.uri = "/b.jpg"

    def test_is_large_file(self):
        from providers.base import MediaContext, MediaModality
        small = MediaContext(uri="/a.jpg", mime="image/jpeg", sha256hash="", modality=MediaModality.IMAGE, file_size=100)
        assert not small.is_large_file
        large = MediaContext(
            uri="/a.mp4", mime="video/mp4", sha256hash="", modality=MediaModality.VIDEO,
            file_size=21 * 1024 * 1024,
        )
        assert large.is_large_file

    def test_default_fields(self):
        from providers.base import MediaContext, MediaModality
        m = MediaContext(uri="/a", mime="image/png", sha256hash="", modality=MediaModality.IMAGE)
        assert m.blob is None
        assert m.pixels is None
        assert m.pair_extras == []
        assert m.extras == {}
        assert m.audio_blob is None


# ──────────────────────────────────────────────
#  PromptContext
# ──────────────────────────────────────────────

class TestPromptContext:

    def test_basic(self):
        from providers.base import PromptContext
        p = PromptContext(system="sys", user="usr")
        assert p.system == "sys"
        assert p.user == "usr"
        assert p.character_name == ""

    def test_with_character(self):
        from providers.base import PromptContext
        p = PromptContext(system="sys", user="describe")
        p2 = p.with_character("Alice", "This is Alice. ")
        assert p2.character_name == "Alice"
        assert p2.user.startswith("This is Alice. ")
        assert "describe" in p2.user
        # 原对象不变
        assert p.user == "describe"

    def test_immutable(self):
        from providers.base import PromptContext
        p = PromptContext(system="s", user="u")
        with pytest.raises(AttributeError):
            p.system = "new"


# ──────────────────────────────────────────────
#  RetryConfig
# ──────────────────────────────────────────────

class TestRetryConfig:

    def test_defaults(self):
        from providers.base import RetryConfig
        r = RetryConfig()
        assert r.max_retries == 10
        assert r.base_wait == 1.0
        assert r.classify_error is None
        assert r.on_exhausted is None

    def test_custom(self):
        from providers.base import RetryConfig
        fn = lambda e: 5.0
        r = RetryConfig(max_retries=3, base_wait=2.0, classify_error=fn)
        assert r.max_retries == 3
        assert r.classify_error is fn


# ──────────────────────────────────────────────
#  ProviderCapabilities
# ──────────────────────────────────────────────

class TestProviderCapabilities:

    def test_defaults(self):
        from providers.capabilities import ProviderCapabilities
        cap = ProviderCapabilities()
        assert not cap.supports_streaming
        assert not cap.supports_structured_output
        assert not cap.supports_audio
        assert not cap.supports_video
        assert not cap.supports_images
        assert not cap.supports_documents
        assert cap.max_file_size_mb == 100
        assert cap.supported_mimes is None

    def test_cloud_vlm_caps(self):
        from providers.cloud_vlm_base import CloudVLMProvider
        assert CloudVLMProvider.capabilities.supports_streaming
        assert CloudVLMProvider.capabilities.supports_video
        assert CloudVLMProvider.capabilities.supports_images
        assert CloudVLMProvider.capabilities.supports_audio

    def test_ocr_caps(self):
        from providers.ocr_base import OCRProvider
        assert OCRProvider.capabilities.supports_documents
        assert OCRProvider.capabilities.supports_images
        assert not OCRProvider.capabilities.supports_streaming


# ──────────────────────────────────────────────
#  PromptResolver
# ──────────────────────────────────────────────

class TestPromptResolver:

    def _config(self, prompts):
        return {"prompts": prompts}

    def test_image_basic(self):
        from providers.resolver import PromptResolver
        config = self._config({
            "system_prompt": "default_sys",
            "image_system_prompt": "image_sys",
            "image_prompt": "describe image",
        })
        resolver = PromptResolver(config, "test_provider")
        args = SimpleNamespace(pair_dir="", gemini_task="")
        p = resolver.resolve("image/jpeg", args)
        assert p.system == "image_sys"
        assert "describe image" in p.user

    def test_video_basic(self):
        from providers.resolver import PromptResolver
        config = self._config({
            "system_prompt": "default_sys",
            "video_system_prompt": "video_sys",
            "video_prompt": "describe video",
        })
        resolver = PromptResolver(config, "test_provider")
        args = SimpleNamespace(pair_dir="")
        p = resolver.resolve("video/mp4", args)
        assert p.system == "video_sys"
        assert "describe video" in p.user

    def test_audio_basic(self):
        from providers.resolver import PromptResolver
        config = self._config({
            "system_prompt": "default_sys",
            "audio_system_prompt": "audio_sys",
            "audio_prompt": "transcribe audio",
        })
        resolver = PromptResolver(config, "test_provider")
        args = SimpleNamespace(pair_dir="")
        p = resolver.resolve("audio/mp3", args)
        assert p.system == "audio_sys"
        assert "transcribe audio" in p.user

    def test_fallback_to_default(self):
        from providers.resolver import PromptResolver
        config = self._config({
            "system_prompt": "default_sys",
            "prompt": "default_prompt",
        })
        resolver = PromptResolver(config, "test_provider")
        args = SimpleNamespace(pair_dir="")
        p = resolver.resolve("image/jpeg", args)
        assert p.system == "default_sys"
        assert p.user == "default_prompt"

    def test_provider_specific_override(self):
        from providers.resolver import PromptResolver
        config = self._config({
            "system_prompt": "default_sys",
            "image_system_prompt": "generic_img_sys",
            "image_prompt": "generic_img_prompt",
            "stepfun_video_system_prompt": "stepfun_vsys",
            "stepfun_video_prompt": "stepfun_vprompt",
        })
        resolver = PromptResolver(config, "stepfun")
        args = SimpleNamespace(pair_dir="")
        p = resolver.resolve("video/mp4", args)
        assert p.system == "stepfun_vsys"
        assert p.user == "stepfun_vprompt"

    def test_pair_mode(self):
        from providers.resolver import PromptResolver
        config = self._config({
            "system_prompt": "sys",
            "image_prompt": "img_prompt",
            "image_pair_system_prompt": "pair_sys",
            "image_pair_prompt": "pair_prompt",
        })
        resolver = PromptResolver(config, "test")
        args = SimpleNamespace(pair_dir="/some/dir")
        p = resolver.resolve("image/jpeg", args)
        assert p.system == "pair_sys"
        assert p.user == "pair_prompt"

    def test_character_prompt_injection(self):
        from providers.resolver import PromptResolver
        config = self._config({
            "system_prompt": "sys",
            "image_prompt": "describe",
        })
        resolver = PromptResolver(config, "test")
        args = SimpleNamespace(pair_dir="")
        p = resolver.resolve("image/jpeg", args, character_prompt="Character is Alice. ", character_name="Alice")
        assert p.user.startswith("Character is Alice. ")
        assert p.character_prompt == "Character is Alice. "
        assert p.character_name == "Alice"

    def test_gemini_task_template(self):
        from providers.resolver import PromptResolver
        config = self._config({
            "system_prompt": "sys",
            "image_prompt": "base",
            "task": {
                "change_a_to_b": "Transform {a} into {b}",
            },
        })
        resolver = PromptResolver(config, "gemini")
        args = SimpleNamespace(pair_dir="", gemini_task="change cat to dog")
        p = resolver.resolve("image/jpeg", args)
        assert "Transform cat into dog" in p.user

    def test_empty_config(self):
        from providers.resolver import PromptResolver
        resolver = PromptResolver({}, "test")
        args = SimpleNamespace(pair_dir="")
        p = resolver.resolve("image/jpeg", args)
        assert p.system == ""
        assert p.user == ""


# ──────────────────────────────────────────────
#  with_retry_impl
# ──────────────────────────────────────────────

class TestWithRetryImpl:

    def test_success_no_retry(self):
        from providers.base import RetryConfig
        from providers.utils import with_retry_impl
        counter = {"n": 0}

        def fn():
            counter["n"] += 1
            return "ok"

        result = with_retry_impl(fn, RetryConfig(max_retries=3, base_wait=0.01))
        assert result == "ok"
        assert counter["n"] == 1

    def test_retry_on_429(self):
        from providers.base import RetryConfig
        from providers.utils import with_retry_impl
        counter = {"n": 0}

        def fn():
            counter["n"] += 1
            if counter["n"] < 3:
                raise Exception("Error 429 rate limited")
            return "ok"

        with patch("providers.utils.time.sleep") as mock_sleep:
            result = with_retry_impl(fn, RetryConfig(max_retries=5, base_wait=0.01))

        assert result == "ok"
        assert counter["n"] == 3
        assert mock_sleep.call_count == 2

    def test_no_retry_on_unknown_error(self):
        from providers.base import RetryConfig
        from providers.utils import with_retry_impl

        def fn():
            raise ValueError("some unknown error")

        with pytest.raises(ValueError, match="some unknown error"):
            with_retry_impl(fn, RetryConfig(max_retries=3, base_wait=0.01))

    def test_custom_classifier(self):
        from providers.base import RetryConfig
        from providers.utils import with_retry_impl
        counter = {"n": 0}

        def classifier(e):
            if "CUSTOM_RETRY" in str(e):
                return 0.01
            return None

        def fn():
            counter["n"] += 1
            if counter["n"] < 2:
                raise Exception("CUSTOM_RETRY please")
            return "done"

        cfg = RetryConfig(max_retries=5, base_wait=0.01, classify_error=classifier)
        result = with_retry_impl(fn, cfg)
        assert result == "done"
        assert counter["n"] == 2

    def test_exhausted_callback(self):
        from providers.base import RetryConfig
        from providers.utils import with_retry_impl

        def always_fail():
            raise Exception("Error 502 server error")

        cfg = RetryConfig(
            max_retries=2,
            base_wait=0.01,
            on_exhausted=lambda e: "fallback_value",
        )
        result = with_retry_impl(always_fail, cfg)
        assert result == "fallback_value"

    def test_exhausted_raises_when_no_callback(self):
        from providers.base import RetryConfig
        from providers.utils import with_retry_impl

        def always_fail():
            raise Exception("Error 502 server error")

        cfg = RetryConfig(max_retries=2, base_wait=0.01)
        with pytest.raises(Exception, match="502"):
            with_retry_impl(always_fail, cfg)

    def test_403_not_retried_by_default(self):
        """默认 classifier 对 403 不重试"""
        from providers.base import RetryConfig
        from providers.utils import with_retry_impl
        counter = {"n": 0}

        def fn():
            counter["n"] += 1
            raise Exception("Error 403 forbidden")

        with pytest.raises(Exception, match="403"):
            with_retry_impl(fn, RetryConfig(max_retries=5, base_wait=0.01))
        assert counter["n"] == 1


# ──────────────────────────────────────────────
#  build_vision_messages
# ──────────────────────────────────────────────

class TestBuildVisionMessages:

    def test_text_first(self):
        from providers.utils import build_vision_messages
        msgs = build_vision_messages("sys", "usr", "base64blob", text_first=True)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "sys"
        content = msgs[1]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert "base64blob" in content[1]["image_url"]["url"]

    def test_text_last(self):
        from providers.utils import build_vision_messages
        msgs = build_vision_messages("sys", "usr", "blob", text_first=False)
        content = msgs[1]["content"]
        assert content[0]["type"] == "image_url"
        assert content[-1]["type"] == "text"

    def test_with_pair(self):
        from providers.utils import build_vision_messages
        msgs = build_vision_messages("sys", "usr", "main_blob", pair_blob="pair_blob", text_first=True)
        content = msgs[1]["content"]
        # text + main image + pair image
        assert len(content) == 3
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "image_url"
        assert "pair_blob" in content[2]["image_url"]["url"]

    def test_no_pair(self):
        from providers.utils import build_vision_messages
        msgs = build_vision_messages("sys", "usr", "blob")
        content = msgs[1]["content"]
        assert len(content) == 2  # text + image only


# ──────────────────────────────────────────────
#  OCRProvider.can_handle 基类逻辑
# ──────────────────────────────────────────────

class TestOCRProviderCanHandle:

    def test_ocr_pdf(self):
        from providers.registry import get_registry
        reg = get_registry()
        cls = reg.get_provider("deepseek_ocr")
        args = SimpleNamespace(ocr_model="deepseek_ocr", document_image=False)
        assert cls.can_handle(args, "application/pdf")

    def test_ocr_image_with_document_image(self):
        from providers.registry import get_registry
        reg = get_registry()
        cls = reg.get_provider("deepseek_ocr")
        args = SimpleNamespace(ocr_model="deepseek_ocr", document_image=True)
        assert cls.can_handle(args, "image/png")

    def test_ocr_image_without_document_image(self):
        from providers.registry import get_registry
        reg = get_registry()
        cls = reg.get_provider("deepseek_ocr")
        args = SimpleNamespace(ocr_model="deepseek_ocr", document_image=False)
        assert not cls.can_handle(args, "image/png")

    def test_ocr_wrong_model_name(self):
        from providers.registry import get_registry
        reg = get_registry()
        cls = reg.get_provider("deepseek_ocr")
        args = SimpleNamespace(ocr_model="hunyuan_ocr", document_image=True)
        assert not cls.can_handle(args, "image/png")

    def test_ocr_video_not_handled(self):
        from providers.registry import get_registry
        reg = get_registry()
        cls = reg.get_provider("deepseek_ocr")
        args = SimpleNamespace(ocr_model="deepseek_ocr", document_image=True)
        assert not cls.can_handle(args, "video/mp4")


# ──────────────────────────────────────────────
#  Provider 类属性完整性
# ──────────────────────────────────────────────

class TestProviderClassAttributes:

    def test_all_providers_have_name(self):
        from providers.registry import get_registry
        reg = get_registry()
        for name in reg.list_providers():
            cls = reg.get_provider(name)
            assert hasattr(cls, "name"), f"{name}: missing .name"
            assert cls.name == name, f"{name}: .name mismatch ({cls.name})"

    def test_all_providers_have_can_handle(self):
        from providers.registry import get_registry
        reg = get_registry()
        for name in reg.list_providers():
            cls = reg.get_provider(name)
            assert hasattr(cls, "can_handle"), f"{name}: missing can_handle"
            assert callable(cls.can_handle), f"{name}: can_handle not callable"

    def test_all_providers_have_attempt(self):
        from providers.registry import get_registry
        reg = get_registry()
        for name in reg.list_providers():
            cls = reg.get_provider(name)
            assert hasattr(cls, "attempt"), f"{name}: missing attempt"

    def test_all_providers_instantiable(self):
        """所有 Provider 都可以正常实例化（使用 mock context），跳过含未实现抽象方法的"""
        import inspect
        from providers.base import ProviderContext
        from providers.registry import get_registry
        from rich.console import Console

        reg = get_registry()
        ctx = ProviderContext(
            console=Console(file=io.StringIO()),
            config={"prompts": {}},
            args=SimpleNamespace(
                max_retries=1, wait_time=0.01, pair_dir="",
                dir_name=False, device=None, dtype=None,
            ),
        )
        for name in reg.list_providers():
            cls = reg.get_provider(name)
            # 跳过仍有未实现抽象方法的类
            if inspect.isabstract(cls):
                continue
            try:
                instance = cls(ctx)
                assert instance is not None
            except TypeError as e:
                if "abstract" in str(e).lower():
                    continue  # 跳过
                pytest.fail(f"Failed to instantiate {name}: {e}")
            except Exception as e:
                pytest.fail(f"Failed to instantiate {name}: {e}")


# ──────────────────────────────────────────────
#  Kimi Code User-Agent header
# ──────────────────────────────────────────────

class TestKimiCodeUserAgent:

    def test_kimi_code_has_user_agent_constant(self):
        from providers.registry import get_registry
        reg = get_registry()
        cls = reg.get_provider("kimi_code")
        assert hasattr(cls, "KIMI_CODE_USER_AGENT")
        assert "claude-code" in cls.KIMI_CODE_USER_AGENT

    def test_kimi_code_openai_client_receives_header(self):
        """验证 OpenAI client 创建时包含 User-Agent header"""
        from providers.base import CaptionResult, MediaContext, MediaModality, PromptContext, ProviderContext
        from providers.registry import get_registry
        from rich.console import Console

        reg = get_registry()
        cls = reg.get_provider("kimi_code")

        ctx = ProviderContext(
            console=Console(file=io.StringIO()),
            config={"kimi_vl": {"thinking": "disabled"}, "prompts": {}},
            args=SimpleNamespace(
                kimi_code_api_key="test-key",
                kimi_code_base_url="https://api.kimi.com/coding/v1",
                kimi_code_model_path="k2p5",
                pair_dir="", mode="long",
                max_retries=1, wait_time=0.01,
            ),
        )
        instance = cls(ctx)

        # Mock OpenAI（在 openai 模块上 mock，因为 kimi_code.py 在 attempt 内部 from openai import OpenAI）
        mock_openai_cls = MagicMock()
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # 确保 kimi_vl_provider 模块在 sys.modules 中（CI 环境可能未预加载）
        import types
        fake_kimi_mod = types.ModuleType("module.providers.kimi_vl_provider")
        fake_kimi_mod.attempt_kimi_vl = MagicMock(return_value="test result")
        saved = sys.modules.get("module.providers.kimi_vl_provider")
        sys.modules["module.providers.kimi_vl_provider"] = fake_kimi_mod
        try:
            with patch("openai.OpenAI", mock_openai_cls):
                media = MediaContext(
                    uri="/fake.jpg", mime="image/jpeg", sha256hash="",
                    modality=MediaModality.IMAGE, blob="base64data", pixels=None,
                )
                prompts = PromptContext(system="sys", user="usr")
                instance.attempt(media, prompts)

            # 验证 OpenAI 创建时有 default_headers
            call_kwargs = mock_openai_cls.call_args
            assert "default_headers" in call_kwargs.kwargs
            assert call_kwargs.kwargs["default_headers"]["User-Agent"] == "claude-code/0.1.0"
        finally:
            if saved is not None:
                sys.modules["module.providers.kimi_vl_provider"] = saved
            else:
                sys.modules.pop("module.providers.kimi_vl_provider", None)

    def test_kimi_code_image_prompt_requests_short_and_long(self):
        from providers.base import MediaContext, MediaModality, PromptContext, ProviderContext
        from providers.registry import get_registry
        from rich.console import Console
        import module.providers.cloud_vlm.kimi_vl as kimi_vl_module

        reg = get_registry()
        cls = reg.get_provider("kimi_code")

        ctx = ProviderContext(
            console=Console(file=io.StringIO()),
            config={"kimi_vl": {"thinking": "disabled"}, "prompts": {}},
            args=SimpleNamespace(
                kimi_code_api_key="test-key",
                kimi_code_base_url="https://api.kimi.com/coding/v1",
                kimi_code_model_path="k2p5",
                pair_dir="", mode="long",
                max_retries=1, wait_time=0.01,
            ),
        )
        instance = cls(ctx)

        captured = {}

        def fake_build_messages(system_prompt, user_prompt, blob, pair_blob=None, text_first=False):
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        with (
            patch("openai.OpenAI", MagicMock(return_value=MagicMock())),
            patch("module.providers.cloud_vlm.kimi_code.build_vision_messages", side_effect=fake_build_messages),
            patch.object(kimi_vl_module, "attempt_kimi_vl", return_value="{}"),
        ):
            media = MediaContext(
                uri="/fake.jpg",
                mime="image/jpeg",
                sha256hash="",
                modality=MediaModality.IMAGE,
                blob="base64data",
                pixels=object(),
            )
            prompts = PromptContext(system="Only output one long paragraph.", user="Describe the image.")
            instance.attempt(media, prompts)

        assert "###Short:" in captured["system_prompt"]
        assert "###Long:" in captured["system_prompt"]
        assert "Do not return only a single long paragraph." in captured["system_prompt"]


class TestKimiStructuredDisplay:

    def test_attempt_kimi_vl_keeps_short_for_display_when_mode_long(self):
        from rich.console import Console
        from module.providers.cloud_vlm.kimi_vl import attempt_kimi_vl

        response_payload = {
            "short_description": "short text",
            "long_description": "long text",
        }
        response_text = json.dumps(response_payload, ensure_ascii=False)

        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=response_text))]
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [chunk]

        with patch("module.providers.cloud_vlm.kimi_vl.display_caption_layout") as mock_display:
            returned = attempt_kimi_vl(
                client=mock_client,
                model_path="kimi-k2.5",
                messages=[],
                console=Console(file=io.StringIO()),
                progress=None,
                task_id=None,
                uri="/fake.jpg",
                image_pixels=object(),
                pair_pixels=None,
                thinking="disabled",
                mode="long",
            )

        assert json.loads(returned) == {"long_description": "long text"}
        assert mock_display.call_args.kwargs["short_description"] == "short text"
        assert mock_display.call_args.kwargs["long_description"] == "long text"


# ──────────────────────────────────────────────
#  Provider 基类逻辑
# ──────────────────────────────────────────────

class TestProviderBase:

    def test_get_retry_config_defaults(self):
        from providers.base import Provider, ProviderContext, CaptionResult, MediaContext, PromptContext
        from rich.console import Console

        class Dummy(Provider):
            name = "dummy"
            @classmethod
            def can_handle(cls, args, mime):
                return True
            def prepare_media(self, uri, mime, args):
                from providers.base import MediaModality
                return MediaContext(uri=uri, mime=mime, sha256hash="", modality=MediaModality.IMAGE)
            def attempt(self, media, prompts):
                return CaptionResult(raw="ok")

        ctx = ProviderContext(
            console=Console(file=io.StringIO()),
            args=SimpleNamespace(max_retries=5, wait_time=2.0),
        )
        d = Dummy(ctx)
        cfg = d.get_retry_config()
        assert cfg.max_retries == 5
        assert cfg.base_wait == 2.0

    def test_log_method(self):
        from providers.base import Provider, ProviderContext, CaptionResult, MediaContext, PromptContext
        from rich.console import Console

        class Dummy(Provider):
            name = "dummy"
            @classmethod
            def can_handle(cls, args, mime):
                return True
            def prepare_media(self, uri, mime, args):
                from providers.base import MediaModality
                return MediaContext(uri=uri, mime=mime, sha256hash="", modality=MediaModality.IMAGE)
            def attempt(self, media, prompts):
                return CaptionResult(raw="ok")

        buf = io.StringIO()
        ctx = ProviderContext(
            console=Console(file=buf, force_terminal=False),
            args=SimpleNamespace(max_retries=1, wait_time=0.01),
        )
        d = Dummy(ctx)
        d.log("test message", "blue")
        d.log("plain message")
        output = buf.getvalue()
        assert "test message" in output
        assert "plain message" in output

    def test_post_validate_default(self):
        from providers.base import Provider, ProviderContext, CaptionResult, MediaContext, MediaModality, PromptContext
        from rich.console import Console

        class Dummy(Provider):
            name = "dummy"
            @classmethod
            def can_handle(cls, args, mime):
                return True
            def prepare_media(self, uri, mime, args):
                return MediaContext(uri=uri, mime=mime, sha256hash="", modality=MediaModality.IMAGE)
            def attempt(self, media, prompts):
                return CaptionResult(raw="ok")

        ctx = ProviderContext(
            console=Console(file=io.StringIO()),
            args=SimpleNamespace(max_retries=1, wait_time=0.01),
        )
        d = Dummy(ctx)
        r = CaptionResult(raw="result")
        m = MediaContext(uri="/a", mime="image/jpeg", sha256hash="", modality=MediaModality.IMAGE)
        assert d.post_validate(r, m, ctx.args) is r

    def test_execute_propagates_sha256hash(self):
        from providers.base import Provider, ProviderContext, CaptionResult, MediaContext, MediaModality
        from rich.console import Console

        captured = {}

        class Dummy(Provider):
            name = "dummy"

            @classmethod
            def can_handle(cls, args, mime):
                return True

            def prepare_media(self, uri, mime, args):
                return MediaContext(uri=uri, mime=mime, sha256hash="", modality=MediaModality.IMAGE)

            def attempt(self, media, prompts):
                captured["sha256hash"] = media.sha256hash
                return CaptionResult(raw="ok")

        ctx = ProviderContext(
            console=Console(file=io.StringIO()),
            config={"prompts": {}},
            args=SimpleNamespace(max_retries=1, wait_time=0.01, dir_name=False, pair_dir="", gemini_task=""),
        )
        Dummy(ctx).execute("/a.jpg", "image/jpeg", "hash-123")
        assert captured["sha256hash"] == "hash-123"


# ──────────────────────────────────────────────
#  OCRProvider 基类方法
# ──────────────────────────────────────────────

class TestOCRProviderBase:

    def test_get_prompts_default(self):
        from providers.base import ProviderContext
        from providers.ocr_base import OCRProvider
        from rich.console import Console

        class FakeOCR(OCRProvider):
            name = "fake_ocr"
            default_prompt = "default ocr prompt"
            def attempt(self, media, prompts):
                pass

        ctx = ProviderContext(
            console=Console(file=io.StringIO()),
            config={"prompts": {}},
            args=SimpleNamespace(max_retries=1, wait_time=0.01),
        )
        p = FakeOCR(ctx)
        sys, usr = p.get_prompts("application/pdf")
        assert sys == ""
        assert usr == "default ocr prompt"

    def test_get_prompts_from_config(self):
        from providers.base import ProviderContext
        from providers.ocr_base import OCRProvider
        from rich.console import Console

        class FakeOCR(OCRProvider):
            name = "fake_ocr"
            default_prompt = "default"
            def attempt(self, media, prompts):
                pass

        ctx = ProviderContext(
            console=Console(file=io.StringIO()),
            config={"prompts": {"fake_ocr_prompt": "config prompt"}},
            args=SimpleNamespace(max_retries=1, wait_time=0.01),
        )
        p = FakeOCR(ctx)
        sys, usr = p.get_prompts("application/pdf")
        assert usr == "config prompt"

    def test_retry_config_always_retries(self):
        from providers.base import ProviderContext
        from providers.ocr_base import OCRProvider
        from rich.console import Console

        class FakeOCR(OCRProvider):
            name = "fake_ocr"
            def attempt(self, media, prompts):
                pass

        ctx = ProviderContext(
            console=Console(file=io.StringIO()),
            config={},
            args=SimpleNamespace(max_retries=3, wait_time=0.5),
        )
        p = FakeOCR(ctx)
        cfg = p.get_retry_config()
        # OCR retry 对任何异常都返回 base_wait
        assert cfg.classify_error(Exception("any error")) == 0.5

    def test_get_model_config(self):
        from providers.base import ProviderContext
        from providers.ocr_base import OCRProvider
        from rich.console import Console

        class FakeOCR(OCRProvider):
            name = "fake_ocr"
            def attempt(self, media, prompts):
                pass

        ctx = ProviderContext(
            console=Console(file=io.StringIO()),
            config={"fake_ocr": {"model_id": "custom-model", "batch_size": 4}},
            args=SimpleNamespace(max_retries=1, wait_time=0.01),
        )
        p = FakeOCR(ctx)
        assert p._get_model_config("model_id") == "custom-model"
        assert p._get_model_config("batch_size") == 4
        assert p._get_model_config("missing_key", "default") == "default"

    def test_execute_uses_provider_specific_prompt(self, tmp_path):
        from providers.base import CaptionResult, ProviderContext
        from providers.ocr_base import OCRProvider
        from rich.console import Console

        doc_path = tmp_path / "sample.pdf"
        doc_path.write_bytes(b"%PDF-1.4")

        class FakeOCR(OCRProvider):
            name = "fake_ocr"
            default_prompt = "default"

            def attempt(self, media, prompts):
                return CaptionResult(raw=prompts.user)

        ctx = ProviderContext(
            console=Console(file=io.StringIO()),
            config={"prompts": {"fake_ocr_prompt": "config prompt", "prompt": "generic prompt"}},
            args=SimpleNamespace(max_retries=1, wait_time=0.01, dir_name=False),
        )
        result = FakeOCR(ctx).execute(str(doc_path), "application/pdf", "pdf-hash")
        assert result.raw == "config prompt"


class TestVisionAPIProviders:

    def test_mistral_prepare_media_uses_vision_api_base_for_images(self, tmp_path):
        from providers.base import MediaModality, ProviderContext
        from module.providers.vision_api.pixtral import MistralOCRProvider
        from rich.console import Console

        image_path = tmp_path / "sample.png"
        image_path.write_bytes(b"image")

        ctx = ProviderContext(
            console=Console(file=io.StringIO()),
            config={},
            args=SimpleNamespace(
                mistral_api_key="mk-xxx",
                pixtral_api_key="",
                max_retries=1,
                wait_time=0.01,
                pair_dir="",
                document_image=False,
            ),
        )

        with patch("providers.vision_api_base.encode_image_to_blob", return_value=("blob", "pixels")):
            media = MistralOCRProvider(ctx).prepare_media(str(image_path), "image/png", ctx.args)

        assert media is not None
        assert media.modality is MediaModality.IMAGE
        assert media.blob == "blob"
        assert media.pixels == "pixels"

    def test_gemini_attempt_uploads_large_media(self):
        from providers.base import MediaContext, MediaModality, PromptContext, ProviderContext
        from module.providers.vision_api.gemini import GeminiProvider
        from rich.console import Console

        ctx = ProviderContext(
            console=Console(file=io.StringIO()),
            config={"generation_config": {"default": {}}},
            args=SimpleNamespace(
                gemini_api_key="gm-xxx",
                gemini_model_path="gemini-2.0-flash",
                gemini_task="",
                pair_dir="",
                max_retries=1,
                wait_time=0.01,
            ),
        )
        provider = GeminiProvider(ctx)
        media = MediaContext(
            uri="/fake.mp4",
            mime="video/mp4",
            sha256hash="sha-123",
            modality=MediaModality.VIDEO,
            file_size=21 * 1024 * 1024,
        )
        prompts = PromptContext(system="sys", user="describe")

        fake_client = MagicMock()
        fake_files = [SimpleNamespace(uri="gs://uploaded-file")]
        with (
            patch("google.genai.Client", return_value=fake_client),
            patch("providers.gemini_utils.upload_or_get", return_value=(True, fake_files)) as mock_upload,
            patch("module.providers.vision_api.gemini.attempt_gemini", return_value="{}") as mock_attempt,
        ):
            provider.attempt(media, prompts)

        mock_upload.assert_called_once()
        assert mock_upload.call_args.kwargs["sha256hash"] == "sha-123"
        assert mock_attempt.call_args.kwargs["files"] == fake_files


# ──────────────────────────────────────────────
#  MediaModality / ProviderType 枚举
# ──────────────────────────────────────────────

class TestEnums:

    def test_media_modality(self):
        from providers.base import MediaModality
        assert MediaModality.IMAGE != MediaModality.VIDEO
        assert MediaModality.AUDIO != MediaModality.DOCUMENT
        assert MediaModality.UNKNOWN is not None

    def test_provider_type(self):
        from providers.base import ProviderType
        assert ProviderType.CLOUD_VLM != ProviderType.LOCAL_VLM
        assert ProviderType.OCR != ProviderType.VISION_API


# ──────────────────────────────────────────────
#  register_provider 装饰器
# ──────────────────────────────────────────────

class TestRegisterProviderDecorator:

    def test_sets_name(self):
        from providers.registry import register_provider

        @register_provider("test_deco_provider")
        class TestProv:
            pass

        assert TestProv.name == "test_deco_provider"

    def test_find_provider_falls_back_to_registered_non_priority_provider(self):
        from providers import registry as registry_module

        class DummyProvider:
            name = "dummy_fallback"

            @classmethod
            def can_handle(cls, args, mime):
                return mime == "image/png"

        reg = registry_module.ProviderRegistry()
        original_providers = reg._providers
        original_priority = reg._priority_order

        try:
            reg._providers = {"dummy_fallback": DummyProvider}
            reg._priority_order = []
            provider = reg.find_provider(SimpleNamespace(), "image/png")
            assert provider is DummyProvider
        finally:
            reg._providers = original_providers
            reg._priority_order = original_priority


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
