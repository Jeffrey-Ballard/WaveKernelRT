# import the relevant pieces
from pathlib import Path
import numpy as np

from wavekernelrt.pipeline.configs import (InputConfig,TStoFFTConfig,FFTtoTSConfig,FFTtoPSDConfig,FFTtoCBFConfig,
                                           PSDAvgConfig)
from wavekernelrt.pipeline.configs import BandEnergyConfig,DMRElementConfig,ShadingConfig,BackgroundEstimateConfig
from wavekernelrt.core.geometry import Geometry
from wavekernelrt.core.steeringvectors_utils import construct_rbe_X3


# Define where the data lives
data_root = Path("/dummy_path/")
event_root = data_root / "EventS5"


# Set the various input files ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# don't get wrapped around the axle about file names here the purpose is to define
# basic processing parameters for each Stage for this dataset. Specific
# run_pipeline_* scripts will overwrite these defaults depending on user goals.
time_series_file = event_root / "dummy_file.ts"
array_type = "line"
soundspeed_ftpers = 4885
sample_rate = 3276.8
DEFAULT_TS_INPUT = InputConfig(file=time_series_file,type="ts")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# TS to FFT Defaults --------------------------------------------------------------
nfft = np.round(sample_rate*2).astype(int)
DEFAULT_TStoFFT = TStoFFTConfig(nfft=np.round(sample_rate*2).astype(int),
                                advance=np.round(sample_rate).astype(int),window_name="tukey")
# ---------------------------------------------------------------------------------

# FFT to TS, initializing without anything means to inherit the details from the input FFT
DEFAULT_FFTtoTS = FFTtoTSConfig()


# FFT to PSD Defaults =============================================================
DEFAULT_FFTtoPSD = FFTtoPSDConfig()
# =================================================================================

# FFT to PSD Defaults =============================================================
integration_period_secs = nfft.astype(float)/sample_rate
DEFAULT_PSDAvg5secs = PSDAvgConfig(time_to_avg_secs=5)
# =================================================================================

# FFT to CBF Defaults *************************************************************
shade = ShadingConfig(window_type="taylor", sll_dB=30.0, nbar=4)

# Form the steer directions @@@@@@@@@@@@@@@@@@@@@@
focus_range_ft = 1000000;
elevation_deg_E = 0;

# There is a slight bow to the swellex arrays, so we form the full 360
# 101 beams which would be ~3x the natural beam width at the design frequency to
# minimize scalloping
Nbeams_oneside = 101;
starboard = np.arccos(np.linspace(1,-1,Nbeams_oneside))*180/np.pi
azimuth_deg_A = np.concatenate((starboard,180+starboard[1:-1])) # Do not include 360 if 0 is already included

steer_rbe_rad_J3 = construct_rbe_X3(focus_range_ft,azimuth_deg_A,elevation_deg_E)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def default_fft_to_cbf(geom: Geometry) -> FFTtoCBFConfig:
    return FFTtoCBFConfig(geom=geom, shading=shade, steer_rbe_rad_J3=steer_rbe_rad_J3)

# *********************************************************************************

# Background Estimate parameters $$$$$$$$$$$$$$$$$$$$$$$$
# Abraham (2019), Sec. 9.3.1, pp. 683--685, motivates repeated,
# nearby-bearing, and nearby-frequency background observations. The median,
# dB-domain smoothing, and clip-and-replace settings below are WaveKernelRT
# choices; see docs/references.md for the citation scope.

bkgd_time_seq_avg = 5
bkgd_sector_beam_size = 10
freq_bin_avg = 20
freq_smoothing_window = np.ones(freq_bin_avg)/freq_bin_avg

car_parms_WP = [(3,3,3),(20,10,5)]

DEFAULT_BGKD_EST = BackgroundEstimateConfig(bkgd_time_seq_avg=bkgd_time_seq_avg,
    bkgd_sector_beam_size=bkgd_sector_beam_size,freq_smoothing_window=freq_smoothing_window,
    car_parms_WP=car_parms_WP,full_circle_flag=True)
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


# Band Energy parameters ============================
# Per-bin background normalization followed by linear aggregation follows
# Abraham (2019), Sec. 9.3, p. 683, Eqs. 9.190--9.191. WaveKernelRT also uses
# bin-width weights to integrate the PSD ratio over hertz.

bands_B2 = [(10,125),(125, 400),(10,400),(300,600),(500,800)]

DEFAULT_ENERGY_DETECT = BandEnergyConfig(bands_B2=bands_B2,background=None,output="float32")
# ============================================

# DMR Element parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dmax = 10
Dmin = 3
wngc_val_dB = -6
wngc_tol_dB = 1
eig_thresh = 0.9
sinr_loss_dB = 1.0

def default_fft_to_dmr(geom: Geometry) -> DMRElementConfig:
    return DMRElementConfig(geom=geom, shading=shade, steer_rbe_rad_J3=steer_rbe_rad_J3,
        Dmax=Dmax, Dmin=Dmin, wngc_val_dB=wngc_val_dB, wngc_tol_dB=wngc_tol_dB, eig_thresh=eig_thresh,
        sinr_loss_dB=sinr_loss_dB)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
