# Energy Detect visualization

Run `example_swellex96_visualization_energy_detect.py` after either Energy
Detect pipeline. Configure the shared output root as described in the
[SWellEx-96 workflow](../../docs/examples/swellex96.md), then match the event,
array, beamformer, background, and output-representation selections.
`num_seqs = -1` loads all remaining records from `first_sequence`; a positive
number limits the records loaded.

Select a band and elevation by zero-based index. The example prints the
available bands and elevations. Float32 records without a background estimate
form a Power BTR; float32 records with a background estimate form an SNR BTR.
Both are stored in decibels. `rank_uint8` records contain relative azimuth
ranks and do not use those additional qualifiers. The example chooses suitable
labels and can display a colorbar for float32 output. The equations, Abraham
(2019) references, and limits of the detector interpretation are documented in
[Band-energy detection](../../docs/concepts/band-energy-detection.md).

## Calling `btr_image`

```python
fig, ax, mesh = analysis_images.btr_image(
    data_TA,                       # Shape: (time records, bearings).
    row_data=time_edges,           # Time vertically.
    col_data=bearings_deg,         # Bearing horizontally.
    row_label="Seconds from first record",
    col_label="Bearing [degrees]",
    cmap="viridis", vmin=None, vmax=None,
    pixels_per_sample=1.0,
)
plt.show()
```

Coordinates can be cell centers or edges. An axis with N samples accepts N
centers or N+1 edges. Omitted coordinates use zero-based centers. A single
center receives a unit-width cell; the example supplies explicit time edges so
a single record still spans its actual duration. Times are relative to the
first stored record's timestamp, not the original recording's start.

The function returns the figure, axes, and mesh for further customization. If
supplying `ax`, use a dedicated figure: its size and axes position will be
changed and automatic layout disabled.

## Logical pixels and Retina displays

`pixels_per_sample` sets the initial data-area size in logical desktop pixels,
with extra fixed margins for labels and a title. `1.0` uses one logical pixel
per time/bearing sample, `2.0` doubles both dimensions, and `0.5` halves them.
On a Retina display the physical rendering can contain two device pixels per
logical pixel.

The sizing applies when the function is called. Window resizing, IDE preview
scaling, `tight_layout`, or backend screen fitting can change it. For a long
recording, select fewer records or lower `pixels_per_sample`.

After drawing, inspect the logical data-area dimensions with:

```python
fig.canvas.draw()
ratio = fig.canvas.device_pixel_ratio
bounds = ax.get_window_extent()
print("Logical data area:", bounds.width / ratio, bounds.height / ratio)
```

Raster exports use the requested output DPI independently of screen pixels.
Vector exports do not have a fixed screen-pixel resolution.
