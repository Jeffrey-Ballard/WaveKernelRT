# General imports
import numpy as np
from pathlib import Path

from wavekernelrt.core.geometry import Geometry
from wavekernelrt.core.shading import Shading
from wavekernelrt.core.beampattern import BeamPattern
from wavekernelrt.pipeline.configs import ShadingConfig

import matplotlib.pyplot as plt

# This script is broken up into 3 sections
# 1. Create the array geometry
# 2. Read & plot the array geometry
# 3. Create a basic shaded CBF beampattern at the design frequency


# ------------------------------------------------------------------
# Create the array geometry
# ------------------------------------------------------------------

# This script assumes a simple sphere geometry where each ring is evenly spaced with an even spacing
# in the z direction

# x is east
# y is north
# z is up (+ is up)

# all array dimensions are assumed to be in inches


# ------------------------------------------------------------------
# User settings — paths and array parameters
# ------------------------------------------------------------------
example_dir = Path(__file__).resolve().parent
array_file = example_dir / "simple_sphere_array.txt"

# Rows will be calculated down below based on the radius
Ncols = 36 # num elements in the in x-y plane around the circle

soundspeed_ftpers = 5000
design_freq_xy = 1000
design_freq_z = 1000

radius_inches = Ncols/(2*np.pi)*(soundspeed_ftpers*12/(2*design_freq_xy))

# The 80 comes from the -40 to 40 in the phi direction below
Nrows = np.ceil(radius_inches*(80*np.pi/180)/(soundspeed_ftpers*12/(2*design_freq_z)))

theta_rad = np.arange(0,360,360/Ncols)*np.pi/180
phi_rad = np.arange(-40,40,80/Nrows)*np.pi/180

THETA_RAD, PHI_RAD = np.meshgrid(theta_rad,phi_rad,indexing='ij')

# 0 deg is north, so sin goes to x, and cos goes to y
xpos = radius_inches * np.sin(THETA_RAD.ravel()) * np.cos(PHI_RAD.ravel())
ypos = radius_inches * np.cos(THETA_RAD.ravel()) * np.cos(PHI_RAD.ravel())
zpos = radius_inches * np.sin(PHI_RAD.ravel())

# Heading is always 0 for sphere arrays


# ------------------------------------------------------------------
# Write the geometry file
# ------------------------------------------------------------------
with open(array_file,"w") as file:
    for x,y,z in zip(xpos, ypos, zpos):
        file.write(f"{x} {y} {z}\n")


# ------------------------------------------------------------------
# Read and plot the geometry
# ------------------------------------------------------------------
geom = Geometry(array_file,array_type="sphere",soundspeed_ftpers=soundspeed_ftpers)
fig_geom, ax_geom =geom.plot_geometry()


# ------------------------------------------------------------------
# Create and plot the beampattern
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# User settings — look direction
# ------------------------------------------------------------------
# set the look direction
range_ft = 1000000 # doesn't matter, looking at far field response for the beampatterns
look_bearing_deg = 210
look_elevation_deg = -20

look_dir_rad_13 = np.array([range_ft,look_bearing_deg*np.pi/180, look_elevation_deg*np.pi/180])


# ------------------------------------------------------------------
# Create and inspect the shading
# ------------------------------------------------------------------
# Get the shading
shade = Shading(geom)

# update the shading with the look direction for cylinder and sphere arrays
shade.update_look_dir(look_dir_rad_13)

# Examples of how to interogate or change the shading
shade.plot_shading("taylor",geom)


# ------------------------------------------------------------------
# Calculate and plot the beampattern
# ------------------------------------------------------------------
# Pull beampattern class
bp = BeamPattern(geom=geom,freq=design_freq_xy)

# plot the beampattern
bp.set_shading_config(ShadingConfig(window_type="taylor", sll_dB=30.0, nbar=4))

fig_bp, ax_im, ax_horz, ax_vert = bp.plot_beampattern_weight_sweep(bp.cbf_weight_QNJ,look_dir_rad_13,freq_display=design_freq_xy)


# ------------------------------------------------------------------
# Display the figures
# ------------------------------------------------------------------
plt.show()
