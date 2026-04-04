import inspect
import importlib
from unittest.mock import patch

import pytest

from tests.provider_v2_helpers import make_provider_args


class TestProviderRegistry:
    def test_singleton(self):
        from providers.registry import ProviderRegistry

        r1 = ProviderRegistry()
        r2 = ProviderRegistry()
        assert r1 is r2

    def test_get_registry(self):
        from providers.registry import get_registry

        reg = get_registry()
        assert reg is not None

    def test_discover_all_22(self):
        from providers.registry import get_registry

        reg = get_registry()
        reg.discover()
        providers = reg.list_providers()
        assert len(providers) >= 25
        expected = [
            "stepfun",
            "ark",
            "qwenvl",
            "glm",
            "kimi_code",
            "kimi_vl",
            "deepseek_ocr",
            "logics_ocr",
            "dots_ocr",
            "qianfan_ocr",
            "lighton_ocr",
            "hunyuan_ocr",
            "glm_ocr",
            "chandra_ocr",
            "olmocr",
            "paddle_ocr",
            "nanonets_ocr",
            "firered_ocr",
            "moondream",
            "qwen_vl_local",
            "step_vl_local",
            "penguin_vl_local",
            "reka_edge_local",
            "acestep_transcriber_local",
            "cohere_transcribe_local",
            "mistral_ocr",
            "gemini",
        ]
        for name in expected:
            assert name in providers, f"Missing provider: {name}"

    def test_registered_providers_are_concrete(self):
        from providers.registry import get_registry

        reg = get_registry()
        reg.discover()
        abstract = [name for name, cls in reg._providers.items() if inspect.isabstract(cls)]
        assert not abstract, f"Abstract providers leaked into registry: {abstract}"

    def test_get_provider_by_name(self):
        from providers.registry import get_registry

        reg = get_registry()
        cls = reg.get_provider("gemini")
        assert cls is not None
        assert cls.name == "gemini"

    def test_get_provider_unknown(self):
        from providers.registry import get_registry

        reg = get_registry()
        assert reg.get_provider("nonexistent_provider") is None

    def test_register_custom(self):
        from providers.base import CaptionResult, MediaContext, Provider
        from providers.registry import get_registry

        class FakeProvider(Provider):
            name = "fake_test"

            @classmethod
            def can_handle(cls, args, mime):
                return getattr(args, "fake", False)

            def prepare_media(self, uri, mime, args):
                from providers.base import MediaModality

                return MediaContext(uri=uri, mime=mime, sha256hash="", modality=MediaModality.IMAGE)

            def attempt(self, media, prompts):
                return CaptionResult(raw="fake result")

        reg = get_registry()
        reg.register("fake_test", FakeProvider)
        assert reg.get_provider("fake_test") is FakeProvider
        reg._providers.pop("fake_test", None)

    def test_discover_strict_raises_with_full_import_traceback(self):
        from providers.registry import ProviderDiscoveryError, get_registry

        reg = get_registry()
        original_providers = dict(reg._providers)
        original_discovered = reg._discovered
        original_failures = dict(getattr(reg, "_import_failures", {}))
        real_import_module = importlib.import_module

        def fake_import_module(name, package=None):
            if name == "module.providers.ocr.paddle":
                raise RuntimeError("boom import")
            return real_import_module(name, package)

        try:
            reg._providers = {}
            reg._discovered = False
            reg._import_failures = {}
            with patch("providers.registry.importlib.import_module", side_effect=fake_import_module):
                with pytest.raises(ProviderDiscoveryError, match="paddle_ocr"):
                    reg.discover(strict=True)

            failure = reg.get_import_failure("paddle_ocr")
            assert failure is not None
            assert "Traceback" in failure.traceback
            assert "RuntimeError: boom import" in failure.traceback
        finally:
            reg._providers = original_providers
            reg._discovered = original_discovered
            reg._import_failures = original_failures

    def test_find_provider_lazy_loads_explicit_audio_provider_before_full_discovery(self, monkeypatch):
        from providers.registry import _PROVIDER_MODULES, get_registry

        reg = get_registry()
        original_providers = dict(reg._providers)
        original_discovered = reg._discovered
        original_failures = dict(getattr(reg, "_import_failures", {}))

        class FakeProvider:
            name = "eureka_audio_local"

            @classmethod
            def can_handle(cls, args, mime):
                return getattr(args, "alm_model", "") == "eureka_audio_local" and mime.startswith("audio")

        calls = []

        def fake_discover_provider_module(provider_name, module_path):
            calls.append((provider_name, module_path))
            reg.register(provider_name, FakeProvider)

        try:
            reg._providers = {}
            reg._discovered = False
            reg._import_failures = {}
            monkeypatch.setattr(reg, "_discover_provider_module", fake_discover_provider_module)
            monkeypatch.setattr(
                reg,
                "discover",
                lambda strict=False: (_ for _ in ()).throw(AssertionError("full discover should not run")),
            )

            provider = reg.find_provider(make_provider_args(alm_model="eureka_audio_local"), "audio/wav")

            assert provider is FakeProvider
            assert calls == [("eureka_audio_local", _PROVIDER_MODULES["eureka_audio_local"])]
        finally:
            reg._providers = original_providers
            reg._discovered = original_discovered
            reg._import_failures = original_failures

    def test_find_provider_lazy_loads_explicit_acestep_audio_provider_before_full_discovery(self, monkeypatch):
        from providers.registry import _PROVIDER_MODULES, get_registry

        reg = get_registry()
        original_providers = dict(reg._providers)
        original_discovered = reg._discovered
        original_failures = dict(getattr(reg, "_import_failures", {}))

        class FakeProvider:
            name = "acestep_transcriber_local"

            @classmethod
            def can_handle(cls, args, mime):
                return getattr(args, "alm_model", "") == "acestep_transcriber_local" and mime.startswith("audio")

        calls = []

        def fake_discover_provider_module(provider_name, module_path):
            calls.append((provider_name, module_path))
            reg.register(provider_name, FakeProvider)

        try:
            reg._providers = {}
            reg._discovered = False
            reg._import_failures = {}
            monkeypatch.setattr(reg, "_discover_provider_module", fake_discover_provider_module)
            monkeypatch.setattr(
                reg,
                "discover",
                lambda strict=False: (_ for _ in ()).throw(AssertionError("full discover should not run")),
            )

            provider = reg.find_provider(make_provider_args(alm_model="acestep_transcriber_local"), "audio/wav")

            assert provider is FakeProvider
            assert calls == [("acestep_transcriber_local", _PROVIDER_MODULES["acestep_transcriber_local"])]
        finally:
            reg._providers = original_providers
            reg._discovered = original_discovered
            reg._import_failures = original_failures

    def test_find_provider_lazy_loads_explicit_cohere_audio_provider_before_full_discovery(self, monkeypatch):
        from providers.registry import _PROVIDER_MODULES, get_registry

        reg = get_registry()
        original_providers = dict(reg._providers)
        original_discovered = reg._discovered
        original_failures = dict(getattr(reg, "_import_failures", {}))

        class FakeProvider:
            name = "cohere_transcribe_local"

            @classmethod
            def can_handle(cls, args, mime):
                return getattr(args, "alm_model", "") == "cohere_transcribe_local" and mime.startswith("audio")

        calls = []

        def fake_discover_provider_module(provider_name, module_path):
            calls.append((provider_name, module_path))
            reg.register(provider_name, FakeProvider)

        try:
            reg._providers = {}
            reg._discovered = False
            reg._import_failures = {}
            monkeypatch.setattr(reg, "_discover_provider_module", fake_discover_provider_module)
            monkeypatch.setattr(
                reg,
                "discover",
                lambda strict=False: (_ for _ in ()).throw(AssertionError("full discover should not run")),
            )

            provider = reg.find_provider(make_provider_args(alm_model="cohere_transcribe_local"), "audio/wav")

            assert provider is FakeProvider
            assert calls == [("cohere_transcribe_local", _PROVIDER_MODULES["cohere_transcribe_local"])]
        finally:
            reg._providers = original_providers
            reg._discovered = original_discovered
            reg._import_failures = original_failures


class TestPriorityOrder:
    def test_all_registered_in_priority(self):
        from providers.registry import get_registry

        reg = get_registry()
        reg.discover()
        registered = set(reg.list_providers())
        priority = set(reg._priority_order)
        missing = registered - priority
        assert not missing, f"Registered but not in priority: {missing}"

    def test_no_phantom_in_priority(self):
        from providers.registry import get_registry

        reg = get_registry()
        reg.discover()
        registered = set(reg.list_providers())
        for name in reg._priority_order:
            assert name in registered, f"Priority entry '{name}' not registered"

    def test_kimi_code_before_kimi_vl(self):
        from providers.registry import get_registry

        reg = get_registry()
        order = reg._priority_order
        assert order.index("kimi_code") < order.index("kimi_vl")


def test_find_provider_returns_none_when_no_provider_matches():
    from providers.registry import get_registry

    provider = get_registry().find_provider(make_provider_args(), "image/jpeg")
    assert provider is None
