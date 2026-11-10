"""Public interface for WaveKernelRT.

The objects named in ``__all__`` form the primary user-facing API. They are
loaded lazily so importing one submodule does not initialize unrelated plotting
and pipeline dependencies.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any


_EXPORTS = {
    "BackgroundEstimateConfig": ("wavekernelrt.pipeline.configs", "BackgroundEstimateConfig"),
    "BandEnergyConfig": ("wavekernelrt.pipeline.configs", "BandEnergyConfig"),
    "BeamPattern": ("wavekernelrt.core.beampattern", "BeamPattern"),
    "BuiltPipeline": ("wavekernelrt.pipeline.configs", "BuiltPipeline"),
    "ChannelCropConfig": ("wavekernelrt.pipeline.configs", "ChannelCropConfig"),
    "DMRElementConfig": ("wavekernelrt.pipeline.configs", "DMRElementConfig"),
    "FFTtoCBFConfig": ("wavekernelrt.pipeline.configs", "FFTtoCBFConfig"),
    "FFTtoPSDConfig": ("wavekernelrt.pipeline.configs", "FFTtoPSDConfig"),
    "FFTtoTSConfig": ("wavekernelrt.pipeline.configs", "FFTtoTSConfig"),
    "FreqCropConfig": ("wavekernelrt.pipeline.configs", "FreqCropConfig"),
    "Geometry": ("wavekernelrt.core.geometry", "Geometry"),
    "InputConfig": ("wavekernelrt.pipeline.configs", "InputConfig"),
    "OutputConfig": ("wavekernelrt.pipeline.configs", "OutputConfig"),
    "PSDAvgConfig": ("wavekernelrt.pipeline.configs", "PSDAvgConfig"),
    "PipelineConfig": ("wavekernelrt.pipeline.configs", "PipelineConfig"),
    "Shading": ("wavekernelrt.core.shading", "Shading"),
    "ShadingConfig": ("wavekernelrt.pipeline.configs", "ShadingConfig"),
    "SteeringVectors": ("wavekernelrt.core.steeringvectors", "SteeringVectors"),
    "TStoFFTConfig": ("wavekernelrt.pipeline.configs", "TStoFFTConfig"),
    "TeeConfig": ("wavekernelrt.pipeline.configs", "TeeConfig"),
    "build_pipeline": ("wavekernelrt.pipeline.build_pipeline", "build_pipeline"),
    "construct_rbe_X3": ("wavekernelrt.core.steeringvectors_utils","construct_rbe_X3"),
    "run_pipeline": ("wavekernelrt.pipeline.run_pipeline", "run_pipeline"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load and cache one public object on first access."""
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
    from wavekernelrt.core.beampattern import BeamPattern
    from wavekernelrt.core.geometry import Geometry
    from wavekernelrt.core.shading import Shading
    from wavekernelrt.core.steeringvectors import SteeringVectors
    from wavekernelrt.core.steeringvectors_utils import construct_rbe_X3
    from wavekernelrt.pipeline.build_pipeline import build_pipeline
    from wavekernelrt.pipeline.configs import (BackgroundEstimateConfig,BandEnergyConfig,BuiltPipeline,
        ChannelCropConfig,DMRElementConfig,FFTtoCBFConfig,FFTtoPSDConfig,FFTtoTSConfig,FreqCropConfig,InputConfig,
        OutputConfig,PipelineConfig,PSDAvgConfig,ShadingConfig,TeeConfig,TStoFFTConfig)
    from wavekernelrt.pipeline.run_pipeline import run_pipeline
