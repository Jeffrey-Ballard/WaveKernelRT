"""Plot and export a short audio snippet from one Swellex96 beam.

The input root comes from example_paths.py. The WAV is written beside the BTS
file; an existing WAV with the same name is overwritten.
Open it in an audio player. No playback library or audio device setup is needed.
Peak normalization changes listening level, so separately normalized snippets
cannot be used to compare absolute beam levels. The plot retains input units.
Audio is difference-filtered to reduce low-frequency noise; the plot remains
the original, unfiltered waveform.
"""

# General imports
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
from example_paths import SWELLEX96_OUTPUT_ROOT
from scipy.signal import lfilter, resample_poly
from scipy.io import wavfile
from wavekernelrt.core.bts_file_utils import bts_read_header, bts_read_sequence


# ------------------------------------------------------------------
# User settings — edit these selections
# ------------------------------------------------------------------
data_root = SWELLEX96_OUTPUT_ROOT
event = "S5"       # "S5" or "S59"
array = "south"    # "north" or "south"

beamformer = "cbf"  # Match the pipeline: "cbf" or "dmr".
beam_index = 0      # Zero-based beam position, not a bearing in degrees.
start_seconds = 30.0  # Seconds from the first sample in the BTS file.
duration_seconds = 10.0
normalize_audio = True
# With normalization disabled, this gain must put samples within [-1, 1].
# Use the same gain for different snippets to preserve relative levels.
audio_gain = 1.0
audio_sample_rate_hz = 16000

match event.lower():
    case "s5":
        input_base = data_root / "EventS5" / f"J1312340.hla.{array.lower()}"
    case "s59":
        input_base = data_root / "EventS59" / f"J1341145.hla.{array.lower()}"

input_file = input_base.with_name(f"{input_base.name}_{beamformer.lower()}_beams.bts")
audio_file = input_file.with_name(f"{input_file.stem}_beam{beam_index}_{start_seconds:g}s_{duration_seconds:g}s.wav")


# ------------------------------------------------------------------
# Read only the records covering the requested snippet
# ------------------------------------------------------------------
header = bts_read_header(input_file)
try:
    sample_rate_hz = float(header["sample_rate_hz"])
    record_length = header["Npt_per_seq"]
    total_samples = header["Nseq"] * record_length
    
    if not 0 <= beam_index < header["Jbeams"]:
        raise ValueError("beam_index is outside the available beam range")
    
    first_sample = round(start_seconds * sample_rate_hz)
    sample_count = round(duration_seconds * sample_rate_hz)
    stop_sample = min(first_sample + sample_count, total_samples)

    if first_sample >= stop_sample:
        raise ValueError("No samples selected; check the start time and duration")

    first_record = first_sample // record_length
    stop_record = (stop_sample + record_length - 1) // record_length
    data_JT, array_headings_deg_T = bts_read_sequence(header,np.arange(first_record,stop_record))
    offset = first_sample - first_record * record_length
    snippet = data_JT[beam_index, offset:offset + stop_sample - first_sample].copy()
finally:
    header["fid"].close()

if not np.all(np.isfinite(snippet)):
    raise ValueError("The selected beam contains NaN or infinite samples")

bearing_deg = np.rad2deg(header["steer_rbe_rad_J3"][beam_index, 1])
actual_duration = snippet.size / sample_rate_hz

print(f"Beam {beam_index}: bearing {bearing_deg:g} degrees")
print(f"Selected {actual_duration:.3f} seconds from {first_sample / sample_rate_hz:.3f} seconds")
if stop_sample < first_sample + sample_count:
    print("Snippet shortened at the end of the file.")


# ------------------------------------------------------------------
# Difference-filter, resample, and write mono 16-bit audio
# ------------------------------------------------------------------
# Combine the two successive-difference filters and apply once at the BTS
# sample rate. Coefficients are [1, -1, -1, 1]; sample count is unchanged.
# Zero initial state can introduce a transient in the first three samples.
difference_filter = np.convolve(np.array([1,-1],dtype=np.float32),np.array([1,0,-1],dtype=np.float32))
filtered_snippet = lfilter(difference_filter, np.array([1], dtype=np.float32), snippet)

# Swellex96's 3276.8 Hz is noninteger. Resampling to 16 kHz preserves pitch
# and duration rather than relabeling the input rate; it adds no bandwidth.
# A bounded rational ratio avoids huge filters from float32 header rounding.
ratio = Fraction(audio_sample_rate_hz / sample_rate_hz).limit_denominator(10000)
audio = resample_poly(filtered_snippet, ratio.numerator, ratio.denominator)
if not np.all(np.isfinite(audio)):
    raise ValueError("Resampling produced nonfinite audio samples")

if normalize_audio:
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak) * 0.95  # Headroom; silent snippets remain silent.
else:
    if not np.isfinite(audio_gain) or audio_gain < 0:
        raise ValueError("audio_gain must be finite and nonnegative")
    
    audio = audio * audio_gain

    if not np.all(np.isfinite(audio)) or np.max(np.abs(audio)) > 1:
        raise ValueError("Reduce audio_gain or enable normalize_audio to avoid clipping")

pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
wavfile.write(audio_file, audio_sample_rate_hz, pcm)
print(f"WAV written to: {audio_file}")
print("Open the WAV in your audio player to listen.")


# ------------------------------------------------------------------
# Plot the original snippet before playback normalization/resampling
# ------------------------------------------------------------------
times_secs = (first_sample + np.arange(snippet.size)) / sample_rate_hz
fig, ax = plt.subplots()
ax.plot(times_secs, snippet)
ax.set_xlabel("Seconds from first BTS sample")
ax.set_ylabel("Amplitude [input sample units]")
ax.set_title(f"Beam {beam_index}: bearing {bearing_deg:g}°")
ax.grid(True)
plt.show()
