import io
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rich.console import Console

from tests.provider_v2_helpers import make_provider_args


class TestApiHandlerV2:
    def test_no_provider_returns_empty(self):
        from module.api_handler_v2 import api_process_batch

        result = api_process_batch(
            uri="/fake.jpg",
            mime="image/jpeg",
            config={"prompts": {}},
            args=make_provider_args(),
            sha256hash="abc",
        )
        assert result.raw == ""

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

    def test_is_v2_enabled(self):
        from module.api_handler_v2 import is_v2_enabled, use_v2

        old_val = os.environ.get("QINGLONG_API_V2", "0")
        try:
            use_v2(True)
            assert is_v2_enabled()
            use_v2(False)
            assert not is_v2_enabled()
        finally:
            os.environ["QINGLONG_API_V2"] = old_val

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


class TestCaptionerV2Switch:
    def test_v1_import_default(self):
        old = os.environ.pop("QINGLONG_API_V2", None)
        try:
            assert os.environ.get("QINGLONG_API_V2", "0") != "1"
        finally:
            if old is not None:
                os.environ["QINGLONG_API_V2"] = old

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
