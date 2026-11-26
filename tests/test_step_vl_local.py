from types import SimpleNamespace

import pytest
from rich.console import Console

from module.providers.base import ProviderContext
from module.providers.local_vlm.step_vl_local import StepVLLocalProvider


@pytest.mark.parametrize(
    ("settings", "expected_model"),
    [({}, "stepfun-ai/Step3-VL-10B"), ({"model_id": "local/step"}, "local/step")],
    ids=["default", "override"],
)
def test_local_runtime_resolves_step_checkpoint(settings, expected_model):
    ctx = ProviderContext(console=Console(), config={"step_vl_local": settings}, args=SimpleNamespace())

    assert StepVLLocalProvider(ctx).get_runtime_backend().model_id == expected_model
