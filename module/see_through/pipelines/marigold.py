from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..extracted.marigold_core import load_marigold_pipeline as _load_marigold_pipeline
from ..extracted.marigold_core import run_marigold_phase

if TYPE_CHECKING:
    from ..cli import SeeThroughRunConfig
    from ..model_manager import SeeThroughModelManager
    from ..runtime import RuntimeContext


def load_marigold_pipeline(
    *,
    repo_id: str,
    runtime_context: "RuntimeContext",
    group_offload: bool = False,
    quant_mode: str = "none",
    console: Any | None = None,
) -> Any:
    return _load_marigold_pipeline(
        repo_id=repo_id,
        runtime_context=runtime_context,
        group_offload=group_offload,
        quant_mode=quant_mode,
        console=console,
    )


class MarigoldPhase:
    def __init__(
        self,
        model_manager: "SeeThroughModelManager",
        config: "SeeThroughRunConfig",
        runtime_context: "RuntimeContext",
        console_obj: Any | None = None,
    ) -> None:
        self.model_manager = model_manager
        self.config = config
        self.runtime_context = runtime_context
        self.console_obj = console_obj

    def run_item(self, source_path: Path, output_dir: Path) -> dict[str, Path]:
        pipeline = self.model_manager.get_marigold_pipeline(
            repo_id=self.config.repo_id_depth,
            group_offload=bool(getattr(self.config, "group_offload", False)),
            quant_mode=str(getattr(self.config, "quant_mode", "none")),
            console=self.console_obj,
        )
        return run_marigold_phase(
            source_path=source_path,
            output_dir=output_dir,
            pipeline=pipeline,
            resolution_depth=int(getattr(self.config, "resolution_depth", 720)),
            inference_steps_depth=int(getattr(self.config, "inference_steps_depth", -1)),
            seed=int(getattr(self.config, "seed", 42)),
        )
