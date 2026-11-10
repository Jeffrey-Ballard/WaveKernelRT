# Array-analysis API

These objects form WaveKernelRT's public geometry, spatial-shading, steering,
and beampattern interface. Application code should import them from
`wavekernelrt`; the implementation paths below are used only so mkdocstrings
can locate their definitions.

::: wavekernelrt.core.geometry.Geometry

## Selected geometry

`Geometry.select_elements()` returns this lightweight geometry description.

::: wavekernelrt.core.geometry.BeamformingGeometry

::: wavekernelrt.core.shading.Shading

::: wavekernelrt.core.steeringvectors.SteeringVectors

::: wavekernelrt.core.beampattern.BeamPattern

## Steering-grid construction

::: wavekernelrt.core.steeringvectors_utils.construct_rbe_X3
