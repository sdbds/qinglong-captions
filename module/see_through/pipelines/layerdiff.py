from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..extracted.layerdiff_core import load_layerdiff_pipeline as _load_layerdiff_pipeline
from ..extracted.layerdiff_core import run_layerdiff_phase

if TYPE_CHECKING:
    from ..cli import SeeThroughRunConfig
    from ..model_manager import SeeThroughModelManager
    from ..runtime import RuntimeContext


def load_layerdiff_pipeline(
    *,
    repo_id: str,
    runtime_context: "RuntimeContext",
    vae_ckpt: str | None = None,
    unet_ckpt: str | None = None,
    group_offload: bool = False,
    quant_mode: str = "none",
    console: Any | None = None,
) -> Any:
    return _load_layerdiff_pipeline(
        repo_id=repo_id,
        runtime_context=runtime_context,
        vae_ckpt=vae_ckpt,
        unet_ckpt=unet_ckpt,
        group_offload=group_offload,
        quant_mode=quant_mode,
        console=console,
    )


class LayerDiffPhase:
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
        pipeline = self.model_manager.get_layerdiff_pipeline(
            repo_id=self.config.repo_id_layerdiff,
            vae_ckpt=getattr(self.config, "vae_ckpt", None),
            unet_ckpt=getattr(self.config, "unet_ckpt", None),
            group_offload=bool(getattr(self.config, "group_offload", False)),
            quant_mode=str(getattr(self.config, "quant_mode", "none")),
            console=self.console_obj,
        )
        return run_layerdiff_phase(
            source_path=source_path,
            output_dir=output_dir,
            pipeline=pipeline,
            resolution=self.config.resolution,
            generator_device=self.runtime_context.device,
            seed=int(getattr(self.config, "seed", 42)),
        )
