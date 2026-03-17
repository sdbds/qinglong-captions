import inspect

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
        assert len(providers) >= 22
        expected = [
            "stepfun",
            "ark",
            "qwenvl",
            "glm",
            "kimi_code",
            "kimi_vl",
            "deepseek_ocr",
            "dots_ocr",
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
