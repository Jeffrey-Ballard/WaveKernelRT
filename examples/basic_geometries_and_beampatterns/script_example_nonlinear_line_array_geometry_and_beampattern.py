# General imports
import numpy as np
from pathlib import Path

from wavekernelrt.core.geometry import Geometry
from wavekernelrt.core.shading import Shading
from wavekernelrt.core.beampattern import BeamPattern

import matplotlib.pyplot as plt

# This script is broken up into 3 sections
# 1. Create the array geometry
# 2. Read & plot the array geometry
# 3. Create a basic shaded CBF beampattern at the design frequency


# ------------------------------------------------------------------
# Create the array geometry
# ------------------------------------------------------------------

# This script assumes a center tapered line.

# x is east
# y is north
# z is up (+ is up)

# all array dimensions are assumed to be in inches


# ------------------------------------------------------------------
# User settings — paths and array parameters
# ------------------------------------------------------------------
example_dir = Path(__file__).resolve().parent
array_file = example_dir / "simple_centertapered_line_array.txt"

# These parameters result in a design freq of 333 Hz for the full aperture and ~5 kHz for the densely
# backed center portion depending on aperture used. A custom prune file would need to be set up and sent
# in to take advantage of center tapered array, but that has not yet been implemented.
Nelements=251 # must be odd for this script so an element lands at zero
Aperture_ft = 500;

r_half_ft = np.logspace(0,np.log10(Aperture_ft/2+1),num=int((Nelements-1)/2)+1)-1 # inches, distance between elements

r_inches = np.concatenate((-1*np.flip(r_half_ft[1:]),r_half_ft))*12

# Can put in an array heading if you want to model a fixed array, but in practice it's usually set to 0
# and bearings shifted from relative to geo afterward
array_heading_rad = 0*np.pi/180 # this is the heading from element 2 to element 1

# 0 deg is north, so sin goes to x, and cos goes to y
# Heading is defined as the direction the 1st element is pointing toward, so the elements need to be
# aligned backward from that here
posx = r_inches*np.sin(array_heading_rad+np.pi)
posy = r_inches*np.cos(array_heading_rad+np.pi)
posz = np.zeros(Nelements)


# ------------------------------------------------------------------
# Write the geometry file
# ------------------------------------------------------------------
with open(array_file,"w") as file:
    [file.write(f"{x} {y} {z}\n") for x,y,z in zip(posx,posy,posz)]


# ------------------------------------------------------------------
# Read and plot the geometry
# ------------------------------------------------------------------
geom = Geometry(array_file,array_type="line",soundspeed_ftpers=5000)
fig_geom, ax_geom = geom.plot_geometry()


# ------------------------------------------------------------------
# User settings — look direction
# ------------------------------------------------------------------
# set the look direction
range_ft = 1000000 # doesn't matter, looking at far field response for the beampatterns
look_bearing_deg = 60
look_elevation_deg = 0

look_dir_rad_13 = np.array([range_ft,look_bearing_deg*np.pi/180, look_elevation_deg*np.pi/180])


# ------------------------------------------------------------------
# Create and inspect the shading
# ------------------------------------------------------------------
# Get the shading
shade = Shading(geom)

# # update the shading with the look direction
# shade.update_look_dir(look_dir_rad_13)

# Examples of how to interogate or change the shading
shade.plot_shading("taylor",geom)


# ------------------------------------------------------------------
# Calculate and plot the beampattern
# ------------------------------------------------------------------
# Pull beampattern class
freq = 400
bp = BeamPattern(geom=geom,freq=freq)

# plot the beampattern
# bp.compute_cbf_weight_QNJ(shade.taylor_shading_N)
bp.compute_cbf_weight_QNJ(shade.uniform_shading_N)

fig_bp,ax_bp = bp.plot_beampattern_weight_sweep(bp.cbf_weight_QNJ,look_dir_rad_13,freq_display=freq)


# ------------------------------------------------------------------
# Display the figures
# ------------------------------------------------------------------
plt.show()
