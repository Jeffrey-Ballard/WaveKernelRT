# General imports
import argparse
import numpy as np
from pathlib import Path

from wavekernelrt.core.geometry import Geometry
from wavekernelrt.core.shading import Shading
from wavekernelrt.core.beampattern import BeamPattern

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Generate a line-array geometry and conventional-beamforming plots.")
parser.add_argument("--output-dir",type=Path,
    help="directory for the generated geometry and PNG figures; display figures when omitted")
args = parser.parse_args()

# This script is broken up into 3 sections
# 1. Create the array geometry
# 2. Read & plot the array geometry
# 3. Create a basic shaded CBF beampattern at the design frequency


# ------------------------------------------------------------------
# Create the array geometry
# ------------------------------------------------------------------

# This script assumes a simple line array that is evenly spaced.

# x is east
# y is north
# z is up (+ is up)

# all array dimensions are assumed to be in inches


# ------------------------------------------------------------------
# User settings — paths and array parameters
# ------------------------------------------------------------------
example_dir = Path(__file__).resolve().parent
output_dir = args.output_dir

if output_dir is not None:
    output_dir.mkdir(parents=True, exist_ok=True)

array_file = (output_dir or example_dir) / "simple_line_array.txt"

design_freq = 1000
soundspeed_ftpers = 5000

Nelements=50
dr = 12*soundspeed_ftpers/(2*design_freq) # inches, distance between elements

# Can put in an array heading if you want to model a fixed array, but in practice it's usually set to 0
# and bearings shifted from relative to geo afterward
array_heading_rad = 0*np.pi/180 # this is the heading from element 2 to element 1

# 0 deg is north, so sin goes to x, and cos goes to y
# Heading is defined as the direction the 1st element is pointing toward, so the elements need to be
# aligned backward from that here
dx = dr*np.sin(array_heading_rad+np.pi)
dy = dr*np.cos(array_heading_rad+np.pi)

if dx==0:
    posx = np.zeros(Nelements)
else:
    posx = np.array(np.arange(0,Nelements*dx,dx))

if dy==0:
    posy = np.zeros(Nelements)
else:
    posy = np.array(np.arange(0,Nelements*dy,dy))

posz = np.zeros(Nelements)


# ------------------------------------------------------------------
# Write the geometry file
# ------------------------------------------------------------------
with open(array_file,"w") as file:
    [file.write(f"{x} {y} {z}\n") for x,y,z in zip(posx,posy,posz)]

# ------------------------------------------------------------------
# Read and plot the geometry
# ------------------------------------------------------------------
geom = Geometry(array_file,array_type="line",soundspeed_ftpers=soundspeed_ftpers)
fig_geom, ax_geom =geom.plot_geometry()


# ------------------------------------------------------------------
# User settings — look direction
# ------------------------------------------------------------------
# set the look direction
range_ft = 1000000 # doesn't matter, looking at far field response for the beampatterns
look_bearing_deg = 90
look_elevation_deg = 0

look_dir_rad_13 = np.array([range_ft,look_bearing_deg*np.pi/180, look_elevation_deg*np.pi/180])


# ------------------------------------------------------------------
# Create and inspect the shading
# ------------------------------------------------------------------
# Get the shading
shade = Shading(geom)

# Examples of how to interogate or change the shading
shade.plot_shading("taylor",geom)
fig_shading = plt.gcf()

# ------------------------------------------------------------------
# Calculate and plot the beampattern
# ------------------------------------------------------------------
# Pull beampattern class
bp = BeamPattern(geom=geom,freq=design_freq)

# plot the beampattern
bp.compute_cbf_weight_QNJ(shade.taylor_shading_N)

fig_weight_sweep,ax_weight_sweep = bp.plot_beampattern_weight_sweep(
    bp.cbf_weight_QNJ,look_dir_rad_13,freq_display=design_freq)

azimuth_deg_A = np.acos(np.linspace(1,-1,101))*180/np.pi
elevation_deg_E = 0
fig_look_sweep,ax_look_sweep = bp.plot_beampattern_look_sweep(
    bp.cbf_weight_QNJ[:,:,10],azimuth_deg_A,elevation_deg_E,freq_display=design_freq)


# ------------------------------------------------------------------
# Save or display the figures
# ------------------------------------------------------------------
if output_dir is None:
    plt.show()
else:
    figures = {
        "line_array_geometry.png": fig_geom,
        "line_array_shading.png": fig_shading,
        "line_array_weight_sweep.png": fig_weight_sweep,
        "line_array_look_sweep.png": fig_look_sweep}
    for filename,figure in figures.items():
        figure.savefig(output_dir / filename)

    print(f"Geometry and figures written to: {output_dir.resolve()}")
