import subprocess
import sys
import textwrap
from pathlib import Path


def test_translate_gui_offers_index_models_and_launches_selected_model(tmp_path):
    code = textwrap.dedent("""\
        import asyncio
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        from gui.path_setup import configure_sys_path
        configure_sys_path(Path.cwd())
        from nicegui import ui
        from wizard.step6_tools import ToolsStep

        calls = []
        async def run_job(script, args, **kwargs):
            calls.append((script, args, kwargs))

        step = ToolsStep()
        step._ensure_execution_panel = lambda: SimpleNamespace(run_job=run_job)
        with ui.column() as container:
            step._render_translate_tool()
        try:
            assert step.translate_model.value == "tencent/Hy-MT2-7B"
            step.translate_input.value = sys.argv[1]
            for model_id in ("IndexTeam/Index-Translate-2B", "IndexTeam/Index-Translate-9B"):
                step.translate_model.set_value(model_id)
                asyncio.run(step._start_translate())
                script, args, kwargs = calls[-1]
                assert script == "module.texttranslate"
                assert f"--model_id={model_id}" in args
                assert "--target_lang=zh_cn" in args
                assert kwargs["runner_kwargs"] == {"uv_extra": "translate"}
        finally:
            container.delete()
        """)
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, encoding="utf-8", timeout=90, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
