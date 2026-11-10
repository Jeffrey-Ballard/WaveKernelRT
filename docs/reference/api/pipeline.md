# Pipeline API

These objects form WaveKernelRT's public declarative pipeline interface. Import
them from `wavekernelrt` in application code.

## Pipeline descriptions

::: wavekernelrt.pipeline.configs.InputConfig

::: wavekernelrt.pipeline.configs.OutputConfig

::: wavekernelrt.pipeline.configs.PipelineConfig

::: wavekernelrt.pipeline.configs.BuiltPipeline

## Spectral transforms and selection

::: wavekernelrt.pipeline.configs.TStoFFTConfig

::: wavekernelrt.pipeline.configs.FFTtoTSConfig

::: wavekernelrt.pipeline.configs.FreqCropConfig

::: wavekernelrt.pipeline.configs.ChannelCropConfig

::: wavekernelrt.pipeline.configs.FFTtoPSDConfig

::: wavekernelrt.pipeline.configs.PSDAvgConfig

## Beamforming

::: wavekernelrt.pipeline.configs.ShadingConfig

::: wavekernelrt.pipeline.configs.FFTtoCBFConfig

::: wavekernelrt.pipeline.configs.DMRElementConfig

## Band-energy processing

::: wavekernelrt.pipeline.configs.BackgroundEstimateConfig

::: wavekernelrt.pipeline.configs.BandEnergyConfig

## Stream output

::: wavekernelrt.pipeline.configs.TeeConfig

## Construction and execution

::: wavekernelrt.pipeline.build_pipeline.build_pipeline

::: wavekernelrt.pipeline.run_pipeline.run_pipeline
