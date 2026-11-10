"""Public configuration and execution interface for WaveKernelRT pipelines."""

from importlib import import_module
from typing import TYPE_CHECKING, Any


_CONFIG_EXPORTS = ("BackgroundEstimateConfig","BandEnergyConfig","BuiltPipeline","ChannelCropConfig",
    "DMRElementConfig","FFTtoCBFConfig","FFTtoPSDConfig","FFTtoTSConfig","FreqCropConfig","InputConfig","OutputConfig",
    "PSDAvgConfig","PipelineConfig","ShadingConfig","StageConfig","TStoFFTConfig","TeeConfig")

_EXPORTS = {
    **{
        name: ("wavekernelrt.pipeline.configs", name)
        for name in _CONFIG_EXPORTS
    },
    "build_pipeline": ("wavekernelrt.pipeline.build_pipeline", "build_pipeline"),
    "run_pipeline": ("wavekernelrt.pipeline.run_pipeline", "run_pipeline"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load and cache one public pipeline object on first access."""
    try:
        module_name, object_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), object_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily exported objects in interactive discovery."""
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from wavekernelrt.pipeline.build_pipeline import build_pipeline
    from wavekernelrt.pipeline.configs import (BackgroundEstimateConfig,BandEnergyConfig,BuiltPipeline,
        ChannelCropConfig,DMRElementConfig,FFTtoCBFConfig,FFTtoPSDConfig,FFTtoTSConfig,FreqCropConfig,InputConfig,
        OutputConfig,PipelineConfig,PSDAvgConfig,ShadingConfig,StageConfig,TeeConfig,TStoFFTConfig)
    from wavekernelrt.pipeline.run_pipeline import run_pipeline
