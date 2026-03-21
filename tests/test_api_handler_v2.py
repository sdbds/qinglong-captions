import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rich.console import Console

from tests.provider_v2_helpers import make_provider_args


class TestApiHandlerV2:
    def test_no_provider_raises_explicit_error(self):
        from module.api_handler_v2 import NoProviderAvailableError, api_process_batch

        with pytest.raises(NoProviderAvailableError, match="mime=image/jpeg") as exc_info:
            api_process_batch(
                uri="/fake.jpg",
                mime="image/jpeg",
                config={"prompts": {}},
                args=make_provider_args(),
                sha256hash="abc",
            )

        message = str(exc_info.value)
        assert "No provider available for mime=image/jpeg" in message
        assert "Available providers:" in message

    def test_no_provider_error_includes_import_failure_summary(self):
        from module.api_handler_v2 import NoProviderAvailableError, api_process_batch

        args = make_provider_args(
            max_retries=1,
            wait_time=0.01,
            dir_name=False,
        )

        class FailedImport:
            def summary(self):
                return "paddle_ocr import failed: RuntimeError: boom"

        class EmptyRegistry:
            def find_provider(self, *_args, **_kwargs):
                return None

            def list_providers(self):
                return ["gemini", "paddle_ocr"]

            def list_import_failures(self):
                return [FailedImport()]

        with (
            patch("module.api_handler_v2.get_registry", return_value=EmptyRegistry()),
            pytest.raises(NoProviderAvailableError) as exc_info,
        ):
            api_process_batch(
                uri="/fake.jpg",
                mime="application/pdf",
                config={"prompts": {}},
                args=args,
                sha256hash="abc",
            )

        message = str(exc_info.value)
        assert "No provider available for mime=application/pdf" in message
        assert "Available providers: ['gemini', 'paddle_ocr']" in message
        assert "Import failures:" in message
        assert "paddle_ocr import failed: RuntimeError: boom" in message

    def test_with_provider_calls_execute(self):
        from providers.base import CaptionResult
        from module.api_handler_v2 import api_process_batch

        args = make_provider_args(
            step_api_key="sk-xxx",
            max_retries=1,
            wait_time=0.01,
            dir_name=False,
        )

        fake_result = CaptionResult(raw="mocked caption")
        with patch("providers.base.Provider.execute", return_value=fake_result):
            result = api_process_batch(
                uri="/fake.jpg",
                mime="image/jpeg",
                config={"prompts": {}},
                args=args,
                sha256hash="abc",
            )
        assert isinstance(result, CaptionResult)
        assert result.raw == "mocked caption"

    def test_gemini_provider_instantiates_through_api_handler(self):
        from providers.base import CaptionResult
        from module.api_handler_v2 import api_process_batch

        args = make_provider_args(
            gemini_api_key="gm-xxx",
            max_retries=1,
            wait_time=0.01,
            dir_name=False,
            gemini_model_path="gemini-2.0-flash",
            gemini_task="",
        )

        fake_result = CaptionResult(raw="mocked gemini")
        with patch("providers.base.Provider.execute", return_value=fake_result):
            result = api_process_batch(
                uri="/fake.jpg",
                mime="image/jpeg",
                config={"prompts": {}, "generation_config": {"default": {}}},
                args=args,
                sha256hash="abc",
            )
        assert result.raw == "mocked gemini"

    def test_api_handler_logs_effective_mistral_provider_name_for_standard_image_mode(self):
        from providers.base import CaptionResult
        from module.api_handler_v2 import api_process_batch
        from module.providers.vision_api.pixtral import MistralOCRProvider

        args = make_provider_args(
            mistral_api_key="mk-xxx",
            max_retries=1,
            wait_time=0.01,
            dir_name=False,
        )
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, color_system=None)

        class StubRegistry:
            def find_provider(self, *_args, **_kwargs):
                return MistralOCRProvider

        with (
            patch("module.api_handler_v2.Console", return_value=console),
            patch("module.api_handler_v2.get_registry", return_value=StubRegistry()),
            patch("providers.base.Provider.execute", return_value=CaptionResult(raw="mocked mistral")),
        ):
            result = api_process_batch(
                uri="/fake.jpg",
                mime="image/jpeg",
                config={"prompts": {}},
                args=args,
                sha256hash="abc",
            )

        assert result.raw == "mocked mistral"
        output = buf.getvalue()
        assert "Using provider: mistral" in output
        assert "Using provider: mistral_ocr" not in output

    def test_api_handler_v2_exports_no_legacy_toggle_helpers(self):
        import module.api_handler_v2 as api_handler_v2

        assert not hasattr(api_handler_v2, "api_process_batch_legacy")
        assert not hasattr(api_handler_v2, "is_v2_enabled")
        assert not hasattr(api_handler_v2, "use_v2")

    def test_provider_failure_logs_full_traceback(self):
        from module.api_handler_v2 import api_process_batch

        args = make_provider_args(
            step_api_key="sk-xxx",
            max_retries=1,
            wait_time=0.01,
            dir_name=False,
        )
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, color_system=None)

        def execute_boom(*_args, **_kwargs):
            raise RuntimeError("execute-fail")

        with (
            patch("module.api_handler_v2.Console", return_value=console),
            patch("providers.base.Provider.execute", side_effect=execute_boom),
            pytest.raises(RuntimeError, match="execute-fail"),
        ):
            api_process_batch(
                uri="/fake.jpg",
                mime="image/jpeg",
                config={"prompts": {}},
                args=args,
                sha256hash="abc",
            )

        output = buf.getvalue()
        assert "Provider" in output
        assert "RuntimeError: execute-fail" in output
        assert "Traceback" in output
        assert "execute_boom" in output

    def test_provider_resolution_failure_logs_full_traceback(self):
        from module.api_handler_v2 import api_process_batch

        args = make_provider_args(
            max_retries=1,
            wait_time=0.01,
            dir_name=False,
        )
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, color_system=None)

        class BrokenRegistry:
            def find_provider(self, *_args, **_kwargs):
                raise RuntimeError("discover-fail")

        with (
            patch("module.api_handler_v2.Console", return_value=console),
            patch("module.api_handler_v2.get_registry", return_value=BrokenRegistry()),
            pytest.raises(RuntimeError, match="discover-fail"),
        ):
            api_process_batch(
                uri="/fake.jpg",
                mime="image/jpeg",
                config={"prompts": {}},
                args=args,
                sha256hash="abc",
            )

        output = buf.getvalue()
        assert "Provider resolution failed" in output
        assert "RuntimeError: discover-fail" in output
        assert "Traceback" in output


class TestCaptionerV2Switch:
    def test_v2_wrapper_unwraps_caption_result(self):
        from providers.base import CaptionResult

        def unwrap(result):
            if hasattr(result, "parsed") and result.parsed is not None:
                return result.parsed
            if hasattr(result, "raw"):
                return result.raw
            return result

        assert unwrap(CaptionResult(raw="plain text")) == "plain text"
        assert unwrap(CaptionResult(raw='{"desc": "hi"}', parsed={"desc": "hi"})) == {"desc": "hi"}
        assert unwrap(CaptionResult(raw="")) == ""
