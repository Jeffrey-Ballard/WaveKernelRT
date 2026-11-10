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

# This script assumes a panel array that is evenly spaced.

# x is east
# y is north
# z is up (+ is up)

# all array dimensions are assumed to be in inches


# ------------------------------------------------------------------
# User settings — paths and array parameters
# ------------------------------------------------------------------
example_dir = Path(__file__).resolve().parent
array_file = example_dir / "simple_panel_array.txt"

design_freq_xy = 1000
design_freq_z = 1000

soundspeed_ftpers = 5000

Nrows = 15 # height, in z plane
Ncols = 20 # width, in x-y plane

drow = soundspeed_ftpers*12/(2*design_freq_z) # inches, distance between adjacent elements in z plane
dcol = soundspeed_ftpers*12/(2*design_freq_xy) # inches, distance between adjacent elements in x-y plane

# Can put in an array heading if you want to model a fixed array, but in practice it's usually set to 0
# and bearings shifted from relative to geo afterward
array_heading_rad = 0*np.pi/180 # this is the heading from element 2 to element 1

# 0 deg is north, so sin goes to x, and cos goes to y
# Heading is defined as the direction the 1st element is pointing toward, so the elements need to be
# aligned backward from that here
colx = np.arange(0,Ncols*dcol,dcol)*np.sin(array_heading_rad+np.pi)
coly = np.arange(0,Ncols*dcol,dcol)*np.cos(array_heading_rad+np.pi)

rowz = np.arange(0,Nrows*drow,drow)


# ------------------------------------------------------------------
# Write the geometry file
# ------------------------------------------------------------------
with open(array_file,"w") as file:
    for x,y in zip(colx, coly):
        for z in rowz:
            file.write(f"{x} {y} {z}\n")

# ------------------------------------------------------------------
# Read and plot the geometry
# ------------------------------------------------------------------
geom = Geometry(array_file,array_type="panel_starboard",soundspeed_ftpers=soundspeed_ftpers)
fig_geom, ax_geom = geom.plot_geometry()


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


# ------------------------------------------------------------------
# Calculate and plot the beampattern
# ------------------------------------------------------------------
# Pull beampattern class
bp = BeamPattern(geom=geom,freq=design_freq_xy)

# plot the beampattern
bp.compute_cbf_weight_QNJ(shade.taylor_shading_N)

fig_bp,ax_im,ax_horz,az_vert = bp.plot_beampattern_weight_sweep(bp.cbf_weight_QNJ,look_dir_rad_13,freq_display=design_freq_xy)


# ------------------------------------------------------------------
# Display the figures
# ------------------------------------------------------------------
plt.show()
