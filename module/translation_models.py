"""Translation model presets shared by the CLI, provider and GUI."""

DEFAULT_HY_MT_MODEL = "tencent/Hy-MT3-7B"

HY_MT_MODELS = (
    DEFAULT_HY_MT_MODEL,
    "tencent/Hy-MT3-1.8B",
    "tencent/Hy-MT3-7B-FP8",
    "tencent/Hy-MT3-1.8B-FP8",
    "tencent/Hy-MT2-7B",
    "tencent/Hy-MT2-1.8B-FP8",
)

INDEX_NAILONG_MODEL = "IndexTeam/Index-Nailong-9B"
INDEX_TRANSLATE_MODELS = (
    "IndexTeam/Index-Translate-9B",
    "IndexTeam/Index-Translate-2B",
    INDEX_NAILONG_MODEL,
)
