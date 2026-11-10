"""Build file-backed runtime pipelines from declarative configurations."""

# import the necessary custom pipeline pieces
from wavekernelrt.pipeline.timebase import Timebase
from wavekernelrt.sources.sources import ExistingFileSourceTS, ExistingFileSourceFFT, ExistingFileSourceBF, ExistingFileSourceBTS
from wavekernelrt.processing.TStoFFT_Stage import TStoFFT_Stage
from wavekernelrt.processing.FFTtoTS_Stage import FFTtoTS_Stage
from wavekernelrt.processing.FFTtoPSD_Stage import FFTtoPSD_Stage
from wavekernelrt.processing.FFTtoCBF_Stage import FFTtoCBF_Stage
from wavekernelrt.processing.FreqCrop_Stage import FreqCrop_Stage
from wavekernelrt.processing.ChannelCrop_Stage import ChannelCrop_Stage
from wavekernelrt.processing.PSDAvg_Stage import PSDAvg_Stage
from wavekernelrt.processing.TeeStage import TeeStage
from wavekernelrt.processing.BFtoBandEnergy_Stage import BFtoBandEnergy_Stage
from wavekernelrt.processing.BeamPSDBackgroundEstimator import BeamPSDBackgroundEstimator
from wavekernelrt.processing.FFTtoDMRElement_Stage import FFTtoDMRElement_Stage
from wavekernelrt.sinks.filesinks import TSFileSink, BTSFileSink, FFTFileSink, BFFileSink, BFWeightFileSink, BTRFileSink, WNGFileSink

# import custom file io
from wavekernelrt.core.ts_file_utils import ts_read_header, ts_write_header
from wavekernelrt.core.bts_file_utils import bts_read_header, bts_write_header
from wavekernelrt.core.fft_file_utils import fft_read_header, fft_write_header
from wavekernelrt.core.bf_file_utils import bf_read_header, bf_write_header
from wavekernelrt.core.bf_weight_file_utils import bf_weight_write_header
from wavekernelrt.core.wng_file_utils import wng_write_header
from wavekernelrt.core.btr_file_utils import btr_write_header

# import classes
from wavekernelrt.core.shading import Shading

# import the configs
from wavekernelrt.pipeline.configs import (PipelineConfig,BuiltPipeline,TStoFFTConfig,FFTtoPSDConfig,
                                           FFTtoCBFConfig,TeeConfig,FreqCropConfig)
from wavekernelrt.pipeline.configs import FFTtoTSConfig, ChannelCropConfig, PSDAvgConfig, BandEnergyConfig, DMRElementConfig

from wavekernelrt.pipeline.blocks import PipelineDataBlock

# import necessary python pieces
import numpy as np
from pathlib import Path
from typing import Any
from scipy.signal.windows import tukey, hann, hamming
from contextlib import ExitStack


def build_pipeline(config: PipelineConfig) -> BuiltPipeline:
    """Validate a pipeline configuration and construct its runtime objects.

    Parameters
    ----------
    config : wavekernelrt.PipelineConfig
        Input description, ordered stage configurations, and final output
        description.

    Returns
    -------
    wavekernelrt.BuiltPipeline
        Open source, ordered runtime stages, and open final sink ready for
        [`run_pipeline`][wavekernelrt.pipeline.run_pipeline.run_pipeline].

    Raises
    ------
    TypeError
        If a stage configuration type is unsupported.
    ValueError
        If file types, extensions, stage ordering, array shapes, numerical
        parameters, or input-derived metadata are incompatible.
    OSError
        If an input cannot be opened or an output cannot be created.

    Notes
    -----
    Building is not side-effect free: it opens the input and creates or
    overwrites the final output and any tee outputs. If construction fails,
    resources opened during the attempt are closed, though an output file
    already created may remain.

    The returned object is not a context manager. Normally pass its fields to
    [`run_pipeline`][wavekernelrt.pipeline.run_pipeline.run_pipeline]; if it
    will not be run, close the source, sink, and any closeable stages manually.

    Examples
    --------
    >>> pipeline = build_pipeline(config)
    >>> run_pipeline(pipeline.source, pipeline.stages, pipeline.sink)
    """

    # build the source, It has it's own internal ExitStack to handle failures
    source, pipeline_data = build_source(config.input)

     # Use ExitStack to clean up the rest if they fail on construction
    with ExitStack() as cleanup:
        cleanup.callback(source.close) # set the clean up action

        stages, pipeline_data = build_stages(config.stages,source,pipeline_data,cleanup=cleanup,
            input_config=config.input,output_config=config.output)
        sink   = build_sink(config.output,config.stages,stages,source,pipeline_data)
        cleanup.callback(sink.close)

        pipeline = BuiltPipeline(source=source,stages=stages,sink=sink)
        cleanup.pop_all() # made it!, remove the failure cleanup callaback, regular cleanup is handled elsewhere
        return pipeline

def build_source(input_config):
    with ExitStack() as cleanup:
        match input_config.type:
            case "ts" | ".ts":

                ts_header = ts_read_header(input_config.file)
                cleanup.callback(ts_header["fid"].close)

                tb = Timebase(origin_sample=0,origin_unix_ns=int(ts_header["start_unix_nanoseconds"]),
                    sample_rate_hz=ts_header["sample_rate_hz"])

                source = ExistingFileSourceTS(header=ts_header,sequence_indices=range(ts_header["Nseq"]))
                source.file_type = "ts"

            case "bts" | ".bts":
            
                bts_header = bts_read_header(input_config.file)
                cleanup.callback(bts_header["fid"].close)

                tb = Timebase(origin_sample=0,origin_unix_ns=int(bts_header["start_unix_nanoseconds"]),
                    sample_rate_hz=bts_header["sample_rate_hz"])

                source = ExistingFileSourceBTS(header=bts_header,sequence_indices=range(bts_header["Nseq"]))
                source.file_type = "bts"

            case "fft" | ".fft":

                fft_header = fft_read_header(input_config.file)
                cleanup.callback(fft_header["fid"].close)

                tb = Timebase(origin_sample=fft_header["nfft"]/2,
                    origin_unix_ns=int(fft_header["start_unix_nanoseconds"]),
                    sample_rate_hz=fft_header["sample_rate_hz"])

                source = ExistingFileSourceFFT(header=fft_header,sequence_indices=range(fft_header["Nseq"]))
                source.file_type = "fft"

            case "bpsd" | ".bpsd" | ".bfft" | "bfft":

                bf_header = bf_read_header(input_config.file)
                cleanup.callback(bf_header["fid"].close)

                tb = Timebase(origin_sample=bf_header["nfft"]/2,
                    origin_unix_ns=int(bf_header["start_unix_nanoseconds"]),sample_rate_hz=bf_header["sample_rate_hz"])
                
                source = ExistingFileSourceBF(header=bf_header,sequence_indices=range(bf_header["Nseq"]))

                match input_config.file.suffix.lower():
                    case ".bfft" | "bfft":
                        source.file_type = "bfft"
                    case ".bpsd" | "bpsd":
                        source.file_type = "bpsd"

            case _:
                raise ValueError(f"Unsupported input type: {input_config.type!r}")   


        pipeline_data = build_source_data_info(source,tb)
        cleanup.pop_all() # made it building the source!

    return source, pipeline_data

def _resolve_dmr_weight_output_file(stage_config, input_config, output_config):
    """Resolve an explicit or derived DMR BFW sidecar path."""

    return _resolve_dmr_sidecar_output_file(stage_config.output_weight_file,".bfw",input_config,output_config,append_to_final=False)


def _resolve_dmr_wng_output_file(stage_config, input_config, output_config):
    """Resolve an explicit or derived DMR WNG sidecar path."""

    return _resolve_dmr_sidecar_output_file(stage_config.output_wng_file,".wng",input_config,output_config,append_to_final=False)


def _resolve_dmr_sidecar_output_file(explicit_file,suffix,input_config,output_config,*,append_to_final):

    if explicit_file is not None:
        return Path(explicit_file)

    final_output_file = Path(output_config.file)
    final_output_type = output_config.type.lower().removeprefix(".")
    if final_output_type in ("bfft", "bpsd"):
        if not append_to_final:
            return final_output_file.with_suffix(suffix)
        return final_output_file.with_name(final_output_file.name + suffix)

    input_stem = Path(input_config.file).stem
    return final_output_file.parent / f"{input_stem}_dmr{suffix}"


def _resolve_dmr_sidecar_band(freq, band_Hz, field_name):
    """Return an inclusive contiguous frequency-index range."""

    if band_Hz is None:
        return 0, freq.size

    band = np.asarray(band_Hz)
    if band.shape != (2,):
        raise ValueError(f"{field_name} must contain two bounds")
    if (not np.issubdtype(band.dtype, np.number)
        or np.iscomplexobj(band)
        or not np.all(np.isfinite(band))):
        raise ValueError(f"{field_name} bounds must be finite real numbers")

    low_hz, high_hz = band
    if low_hz > high_hz:
        raise ValueError(f"Lower {field_name} bound must not exceed the upper bound")
    if low_hz < freq[0] or high_hz > freq[-1]:
        raise ValueError(
            f"{field_name} bounds must lie within the current spectrum: "
            f"{freq[0]} to {freq[-1]} Hz")

    idx0 = int(np.searchsorted(freq, low_hz, side="left"))
    idx1 = int(np.searchsorted(freq, high_hz, side="right"))
    if idx0 == idx1:
        raise ValueError(f"Requested {field_name} contains no FFT bins")

    return idx0, idx1


def build_stages(stage_configs,source,pipeline_data,*,cleanup: ExitStack,input_config,output_config):

    header = source.header
    stages: list[Any] = []

    stage_idx = 0
    for stage_config in stage_configs:
        if isinstance(stage_config,ChannelCropConfig):
            # Validate the input data coming in
            if pipeline_data.file_type not in ("ts", "fft", "psd"):
                raise ValueError(
                    f"ChannelCropConfig at stage {stage_idx} requires "
                    f"'ts', 'fft', or 'psd' input; "
                    f"received {pipeline_data.file_type!r}")

            # Validate the channel indexes to crop
            indices = np.asarray(stage_config.idx_channels_to_keep)

            if indices.ndim != 1 or indices.size == 0:
                raise ValueError("Channel indices must be a nonempty 1-D sequence")

            if not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("Channel indices must be integers")

            nchannels = len(pipeline_data.channels)

            if np.any((indices < 0) | (indices >= nchannels)):
                raise ValueError(f"Channel indices must be between 0 and {nchannels - 1}")

            # Update the pipeline so anything downstream knows to crop channels in necessary (geom & shading for beamforming)
            pipeline_data.channels = pipeline_data.channels[indices]
            pipeline_data.source_channel_indices = (pipeline_data.source_channel_indices[indices])
            pipeline_data.Nchan = len(pipeline_data.channels)

            # append the stage the processing string
            stages.append(ChannelCrop_Stage(idx_channels_to_keep=indices,file_type=pipeline_data.file_type))

        elif isinstance(stage_config,TStoFFTConfig):

            # Validate the input data coming in
            match pipeline_data.file_type:
                case "ts":
                    pipeline_data.file_type = "fft"
                case "bts":
                    pipeline_data.file_type = "bfft"
                case _:
                    raise ValueError(
                        f"TStoFFTConfig at stage {stage_idx} requires "
                        f"'ts' or 'bts' input; "
                        f"received {pipeline_data.file_type!r}")
     
            overlap = 1-stage_config.advance/stage_config.nfft

            match stage_config.window_name.lower():
                case "tukey":
                    window_T = tukey(stage_config.nfft,alpha=overlap)
                case "boxcar":
                    window_T = np.ones(stage_config.nfft, dtype=np.float32)
                case "hann" | "hanning":
                    window_T = hann(stage_config.nfft, sym=True).astype(np.float32)
                case "hamming":
                    window_T = hamming(stage_config.nfft, sym=True).astype(np.float32)
                case _:
                    raise ValueError(f"Unsupported FFT window: {stage_config.window_name!r}")

            fft_stage = TStoFFT_Stage(nfft=stage_config.nfft,advance=stage_config.advance,sample_rate_hz=header["sample_rate_hz"],
                file_type=pipeline_data.file_type,window_T=window_T)

            pipeline_data.nfft = stage_config.nfft
            pipeline_data.advance = stage_config.advance
            pipeline_data.freq = np.fft.rfftfreq(pipeline_data.nfft, d=1/pipeline_data.sample_rate_hz)
            # Store the exact power-normalized float32 window applied by the stage.
            pipeline_data.window_T = fft_stage.window_T

            # update the sample and start time
            pipeline_data.start_time_ns = pipeline_data.sample_to_unix_ns(stage_config.nfft/2)
            
            stages.append(fft_stage)

        elif isinstance(stage_config,FFTtoTSConfig):

            # Validate the input data coming in
            match pipeline_data.file_type:
                case "fft":
                    pipeline_data.file_type = "ts"
                case "bfft":
                    pipeline_data.file_type = "bts"
                case _:
                    raise ValueError(
                        f"FFTtoTSConfig at stage {stage_idx} requires "
                        f"'fft' or 'bfft' input; "
                        f"received {pipeline_data.file_type!r}")
                
            if stage_config.nfft is not None:
                pipeline_data.nfft = stage_config.nfft

            if stage_config.advance is not None:
                pipeline_data.advance = stage_config.advance

            if stage_config.Npt_per_seq is None:
                pipeline_data.Npt_per_seq = pipeline_data.advance
            else:
                pipeline_data.Npt_per_seq = stage_config.Npt_per_seq

            if stage_config.window_T is not None:
                pipeline_data.window_T = stage_config.window_T

            # adjust to set the start time back to sample 0 for time series
            start_sample = pipeline_data.unix_ns_to_sample(pipeline_data.start_time_ns) - pipeline_data.nfft/2  
            pipeline_data.start_time_ns = pipeline_data.sample_to_unix_ns(start_sample)

            stages.append(
                FFTtoTS_Stage(freq=pipeline_data.freq,nfft=pipeline_data.nfft,advance=pipeline_data.advance,
                    sample_rate_hz=pipeline_data.sample_rate_hz,Npt_per_seq=pipeline_data.Npt_per_seq,
                    window_T=pipeline_data.window_T))

        elif isinstance(stage_config,FreqCropConfig):

            # Validate the input data coming in
            if pipeline_data.file_type not in ("fft", "psd", "bfft", "bpsd"):
                raise ValueError(
                    f"FreqCropConfig at stage {stage_idx} requires "
                    f"'fft', 'psd', 'bfft', or 'bpsd' input; "
                    f"received {pipeline_data.file_type!r}")

            # validate the frequencies to crop
            band = np.asarray(stage_config.band_to_keep_Hz)

            if band.shape != (2,):
                raise ValueError("band_to_keep_Hz must contain two bounds")

            if (not np.issubdtype(band.dtype, np.number)
                or np.iscomplexobj(band)):
                raise ValueError("Frequency bounds must be real numbers")

            if not np.all(np.isfinite(band)):
                raise ValueError("Frequency bounds must be finite")

            low_hz, high_hz = band

            if low_hz > high_hz:
                raise ValueError("Lower frequency bound must not exceed upper bound")

            freq = pipeline_data.freq

            if low_hz < freq[0] or high_hz > freq[-1]:
                raise ValueError(
                    f"Frequency bounds must lie within the current spectrum: "
                    f"{freq[0]} to {freq[-1]} Hz")

            crop_stage = FreqCrop_Stage(band_to_keep_Hz=stage_config.band_to_keep_Hz,freq=pipeline_data.freq,
                file_type=pipeline_data.file_type)
            stages.append(crop_stage)

            # need to update the pipeline data with the cropped frequency coming out of here
            pipeline_data.freq = pipeline_data.freq[crop_stage.idx0:crop_stage.idx1]
            
        elif isinstance(stage_config,FFTtoPSDConfig):

            # what are we converting, keep track in the pipeline even though FFTtoPSD doesn't need the information specifically
            match pipeline_data.file_type:
                case "fft" | ".fft":
                    pipeline_data.file_type = "psd"
                case "bfft" | ".bfft":
                    pipeline_data.file_type = "bpsd"
                case _:
                    raise ValueError(
                        f"FFTtoPSDConfig at stage {stage_idx} requires "
                        f"'fft' or 'bfft' input; "
                        f"received {pipeline_data.file_type!r}")

            stages.append(FFTtoPSD_Stage(integration_period_secs=pipeline_data.nfft / pipeline_data.sample_rate_hz))

        elif isinstance(stage_config,TeeConfig):

            tee_sink = build_sink(stage_config.output,stage_configs,stages,source,pipeline_data)
            cleanup.callback(tee_sink.close)

            stages.append(TeeStage(sink=tee_sink))

        elif isinstance(stage_config,FFTtoCBFConfig):

            # Validate the input data coming in
            if pipeline_data.file_type != "fft":
                raise ValueError(
                    f"FFTtoCBFConfig at stage {stage_idx} requires "
                    f"'fft' input; "
                    f"received {pipeline_data.file_type!r}")

            nchannels = len(pipeline_data.channels)
            if nchannels < 2:
                raise ValueError(
                    f"FFTtoCBFConfig at stage {stage_idx} "
                    f"requires at least two channels; received {nchannels}")

            nelements = stage_config.geom.xyz_m_N3.shape[0]
            source_count = pipeline_data.source_channel_count

            if nelements != source_count:
                raise ValueError(
                    f"FFTtoCBFConfig at stage {stage_idx}: "
                    f"geometry has {nelements} elements, "
                    f"but the source has {source_count} channels")

            # Downselect the geometry if any channels have been cropped coming into here
            geom = stage_config.geom.select_elements(pipeline_data.source_channel_indices)

            # make sure we didn't crop something without cropping the geometry as well
            if geom.xyz_m_N3.shape[0] != nchannels:
                raise ValueError("Selected geometry does not match the current channel count")

            # Now compute the shade for the geometry actually used 
            shading = Shading(geom)
            is_curved = geom.array_type in ("sphere", "cylinder")
            beams = stage_config.steer_rbe_rad_J3

            match stage_config.shading.window_type:
                case "uniform":
                    if is_curved:
                        shade = shading.uniform_shading_for_beams(beams, flag_relative=True)
                    else:
                        shade = shading.uniform_shading_N.reshape(-1)

                case "hamming":
                    if is_curved:
                        shade = shading.hamming_shading_for_beams(beams, flag_relative=True)
                    else:
                        shade = shading.hamming_shading_N.reshape(-1)

                case "taylor":
                    shading.set_taylor_params(sll_dB=stage_config.shading.sll_dB,nbar=stage_config.shading.nbar)

                    if is_curved:
                        shade = shading.taylor_shading_for_beams(beams, flag_relative=True)
                    else:
                        shade = shading.taylor_shading_N.reshape(-1)

                case _:
                    raise ValueError("Unsupported shading window type")

            # capture what this stage is using in case we want to tee out or write out to a file from here
            pipeline_data.file_type = "bfft"
            pipeline_data.shade_N = None if is_curved else shade
            pipeline_data.shade_JN = shade if is_curved else None
            pipeline_data.steer_rbe_rad_J3 = stage_config.steer_rbe_rad_J3

            stages.append(
                FFTtoCBF_Stage(geom=geom,shade=shade,steer_rbe_rad_J3=stage_config.steer_rbe_rad_J3,
                            freq_process=pipeline_data.freq,
                            heading_provider=stage_config.heading_provider))
            
        elif isinstance(stage_config,PSDAvgConfig):

            # Validate the input data coming in
            if pipeline_data.file_type not in ("psd", "bpsd"):
                raise ValueError(
                    f"PSDAvgConfig at stage {stage_idx} requires "
                    f"'psd' or 'bpsd' input; "
                    f"received {pipeline_data.file_type!r}")

            if stage_config.time_to_avg_secs is not None:
                seq_to_avg = np.round(stage_config.time_to_avg_secs / (pipeline_data.advance/pipeline_data.sample_rate_hz)).astype('int')
            else:
                seq_to_avg = stage_config.seq_to_avg

            if seq_to_avg < 1:
                raise ValueError("Requested averaging duration rounds to fewer than one input record")

            # Update sample and time start
            # Shift the first output timestamp to the midpoint of the input centers.
            # Use the input advance before updating it for the averaged output.
            first_center_sample = pipeline_data.unix_ns_to_sample(pipeline_data.start_time_ns)
            
            average_center_sample = first_center_sample + (seq_to_avg - 1) * pipeline_data.advance / 2

            pipeline_data.start_time_ns = pipeline_data.sample_to_unix_ns(average_center_sample)
            pipeline_data.advance *= seq_to_avg

            stages.append(PSDAvg_Stage(seq_to_avg=seq_to_avg))

        elif isinstance(stage_config, BandEnergyConfig):

            if pipeline_data.file_type != "bpsd":
                raise ValueError(
                    f"BandEnergyConfig at stage {stage_idx} requires "
                    f"'bpsd' input; "
                    f"received {pipeline_data.file_type!r}")

            nbeams = pipeline_data.steer_rbe_rad_J3.shape[0]
            background_estimator = None
            if stage_config.background is not None:
                background = stage_config.background
                background_estimator = BeamPSDBackgroundEstimator(nbeams=nbeams,
                    bkgd_time_seq_avg=background.bkgd_time_seq_avg,
                    bkgd_sector_beam_size=(background.bkgd_sector_beam_size),
                    full_circle_flag=background.full_circle_flag,
                    freq_smoothing_window=background.freq_smoothing_window,car_parms_WP=background.car_parms_WP)

                if background.bkgd_time_seq_avg is not None:
                    first_input_sample = pipeline_data.unix_ns_to_sample(pipeline_data.start_time_ns)
                    center_offset_samples = (
                        background.bkgd_time_seq_avg // 2
                    ) * pipeline_data.advance
                    pipeline_data.start_time_ns = (pipeline_data.sample_to_unix_ns(first_input_sample + center_offset_samples))

            band_energy_stage = BFtoBandEnergy_Stage(freq=pipeline_data.freq,bands_B2=stage_config.bands_B2,
                steer_rbe_rad_J3=pipeline_data.steer_rbe_rad_J3,output=stage_config.output,
                background_estimator=background_estimator)
            stages.append(band_energy_stage)

            pipeline_data.file_type = "btr"
            pipeline_data.bands_B2 = band_energy_stage.bands_B2
            pipeline_data.btr_azim_deg_A = band_energy_stage.azim_deg_A
            pipeline_data.btr_elev_deg_E = band_energy_stage.elev_deg_E
            pipeline_data.btr_dtype = np.dtype(np.float32 if stage_config.output == "float32" else np.uint8)
            
        elif isinstance(stage_config,DMRElementConfig):

            # Validate the input data coming in
            if pipeline_data.file_type != "fft":
                raise ValueError(
                    f"DMRElementConfig at stage {stage_idx} requires "
                    f"'fft' input; "
                    f"received {pipeline_data.file_type!r}")

            nchannels = len(pipeline_data.channels)
            if nchannels < 2:
                raise ValueError(
                    f"DMRElementConfig at stage {stage_idx} "
                    f"requires at least two channels; received {nchannels}")

            nelements = stage_config.geom.xyz_m_N3.shape[0]
            source_count = pipeline_data.source_channel_count

            if nelements != source_count:
                raise ValueError(
                    f"DMRElementConfig at stage {stage_idx}: "
                    f"geometry has {nelements} elements, "
                    f"but the source has {source_count} channels")

            # Downselect the geometry if any channels have been cropped coming into here
            geom = stage_config.geom.select_elements(pipeline_data.source_channel_indices)

            # make sure we didn't crop something without cropping the geometry as well
            if geom.xyz_m_N3.shape[0] != nchannels:
                raise ValueError("Selected geometry does not match the current channel count")

            # Now compute the shade for the geometry actually used 
            shading = Shading(geom)
            is_curved = geom.array_type in ("sphere", "cylinder")
            beams = stage_config.steer_rbe_rad_J3

            match stage_config.shading.window_type:
                case "uniform":
                    if is_curved:
                        shade = shading.uniform_shading_for_beams(beams, flag_relative=True)
                    else:
                        shade = shading.uniform_shading_N.reshape(-1)

                case "hamming":
                    if is_curved:
                        shade = shading.hamming_shading_for_beams(beams, flag_relative=True)
                    else:
                        shade = shading.hamming_shading_N.reshape(-1)

                case "taylor":
                    shading.set_taylor_params(sll_dB=stage_config.shading.sll_dB,nbar=stage_config.shading.nbar)

                    if is_curved:
                        shade = shading.taylor_shading_for_beams(beams, flag_relative=True)
                    else:
                        shade = shading.taylor_shading_N.reshape(-1)

                case _:
                    raise ValueError("Unsupported shading window type")
            
            pipeline_data.file_type = "bfft"
            pipeline_data.shade_N = None if is_curved else shade
            pipeline_data.shade_JN = shade if is_curved else None
            pipeline_data.steer_rbe_rad_J3 = beams

            freq = np.asarray(pipeline_data.freq)
            output_weight_file = None
            output_wng_file = None
            if stage_config.write_weights:
                output_weight_file = _resolve_dmr_weight_output_file(stage_config,input_config,output_config)
                if output_weight_file.suffix.lower() != ".bfw":
                    raise ValueError("DMR output_weight_file must use the .bfw extension")

            if stage_config.write_wng:
                output_wng_file = _resolve_dmr_wng_output_file(stage_config,input_config,output_config)
                if output_wng_file.suffix.lower() != ".wng":
                    raise ValueError("DMR output_wng_file must use the .wng extension")

            if (output_weight_file is not None
                and output_wng_file is not None
                and output_weight_file == output_wng_file):
                raise ValueError("DMR weight and WNG output files must be different")

            weight_sink = None
            weight_freq_range = None
            if output_weight_file is not None:
                weight_idx0, weight_idx1 = _resolve_dmr_sidecar_band(freq,stage_config.weight_band_Hz,"weight_band_Hz")

                weight_header = bf_weight_write_header(output_weight_file,Nseq=0,
                    sample_rate_hz=pipeline_data.sample_rate_hz,start_unix_nanoseconds=pipeline_data.start_time_ns,
                    nfft=pipeline_data.nfft,advance=(pipeline_data.advance * stage_config.weight_seq_stride),
                    freq=freq[weight_idx0:weight_idx1],channels=pipeline_data.channels,steer_rbe_rad_J3=beams,
                    window_T=pipeline_data.window_T,dtype=np.complex64)
                weight_sink = BFWeightFileSink(weight_header)
                cleanup.callback(weight_sink.close)
                weight_freq_range = (weight_idx0, weight_idx1)

            wng_sink = None
            wng_freq_range = None
            if output_wng_file is not None:
                wng_idx0, wng_idx1 = _resolve_dmr_sidecar_band(freq,stage_config.wng_band_Hz,"wng_band_Hz")
                wng_header = wng_write_header(output_wng_file,Nseq=0,sample_rate_hz=pipeline_data.sample_rate_hz,
                    start_unix_nanoseconds=pipeline_data.start_time_ns,nfft=pipeline_data.nfft,
                    advance=(pipeline_data.advance * stage_config.wng_seq_stride),freq=freq[wng_idx0:wng_idx1],
                    steer_rbe_rad_J3=beams,window_T=pipeline_data.window_T,shade=shade,
                    wngc_val_dB=stage_config.wngc_val_dB,wngc_tol_dB=stage_config.wngc_tol_dB)
                wng_sink = WNGFileSink(wng_header)
                cleanup.callback(wng_sink.close)
                wng_freq_range = (wng_idx0, wng_idx1)

            stages.append(
                FFTtoDMRElement_Stage(geom=geom,shade=shade,steer_rbe_rad_J3=stage_config.steer_rbe_rad_J3,freq_process=pipeline_data.freq,
                                    Dmax=stage_config.Dmax,Dmin=stage_config.Dmin,wngc_val_dB=stage_config.wngc_val_dB,
                                    wngc_tol_dB=stage_config.wngc_tol_dB,eig_thresh=stage_config.eig_thresh,sinr_loss_dB=stage_config.sinr_loss_dB,
                                    write_weights=stage_config.write_weights,weight_sink=weight_sink,weight_freq_range=weight_freq_range,
                                    write_wng=stage_config.write_wng,wng_sink=wng_sink,wng_freq_range=wng_freq_range,weight_seq_stride=stage_config.weight_seq_stride,
                                    wng_seq_stride=stage_config.wng_seq_stride,heading_provider=stage_config.heading_provider)
                        )

        else:
            raise ValueError(
                f"Unsupported stage configuration at index {stage_idx}: "
                f"{type(stage_config).__name__}")

        stage_idx += 1          

    return stages, pipeline_data

def build_sink(output_config,stage_configs,stages,source,pipeline_data):

    # Verify that the user set up the OutputConfig properly
    output_type = output_config.type.lower().removeprefix(".")
    extension = output_config.file.suffix.lower().removeprefix(".")

    if extension != output_type:
        raise ValueError(
            f"Output type {output_config.type!r} does not match "
            f"filename extension {output_config.file.suffix!r}")

    if output_type in ("bfft", "bpsd", "bts"):
        if ((pipeline_data.shade_N is None)== (pipeline_data.shade_JN is None)):
            raise ValueError("Beamformed output requires exactly one shading representation")

        shade = pipeline_data.shade_JN if pipeline_data.shade_JN is not None else pipeline_data.shade_N

    pipeline_type = pipeline_data.file_type

    if pipeline_type is None:
        raise ValueError("Pipeline output data type is not defined")

    pipeline_type = pipeline_type.lower().removeprefix(".")

    if output_type != pipeline_type:
        raise ValueError(
            f"Pipeline produces {pipeline_type!r} data, "
            f"but output requests {output_type!r}. "
            "Add the appropriate conversion stage.")

    match output_type:
        case ".ts" | "ts":
            Nseq = 0

            # make the header
            ts_header = ts_write_header(output_config.file,Nseq,pipeline_data.sample_rate_hz,pipeline_data.start_time_ns,
                                          pipeline_data.Npt_per_seq,pipeline_data.channels)

            try:
                return TSFileSink(header=ts_header, check_finite=output_config.check_finite)
            except BaseException:
                ts_header["fid"].close()
                raise

        case ".bts" | "bts":
            Nseq = 0

            # make the header
            bts_header = bts_write_header(output_config.file,Nseq,pipeline_data.sample_rate_hz,pipeline_data.start_time_ns,
                                            pipeline_data.Npt_per_seq,shade,pipeline_data.steer_rbe_rad_J3)
            try:
                return BTSFileSink(header=bts_header, check_finite=output_config.check_finite)
            except BaseException:
                bts_header["fid"].close()
                raise

        case ".fft" | "fft" | ".psd" | "psd":
            match output_config.file.suffix.lower():
                case ".fft" | "fft":
                    dtype = np.complex64
                case ".psd" | "psd":
                    dtype = np.float32

            # compute some derived parameters for the header
            Nseq   = 0; # will go back and overwrite the file with the correct value, will not know this from live real time input

            # make the header
            fft_header = fft_write_header(output_config.file,Nseq,pipeline_data.sample_rate_hz,
                pipeline_data.start_time_ns,pipeline_data.nfft,pipeline_data.advance,pipeline_data.freq,
                pipeline_data.channels,pipeline_data.window_T,dtype)

            try:
                return FFTFileSink(header=fft_header, check_finite=output_config.check_finite)
            except BaseException:
                fft_header["fid"].close()
                raise
        
        case ".bpsd" | "bpsd" | ".bfft" | "bfft":

            match output_config.file.suffix.lower():
                case ".bfft" | "bfft":
                    dtype = np.complex64
                case ".bpsd" | "bpsd":
                    dtype = np.float32

            # write the header for the file out
            Nseq=0 #will be overwritten when the file is closed
    
            bf_header = bf_write_header(output_config.file,Nseq,pipeline_data.sample_rate_hz,pipeline_data.start_time_ns,pipeline_data.nfft,
                                        pipeline_data.advance,pipeline_data.freq,pipeline_data.steer_rbe_rad_J3,pipeline_data.window_T,
                                        shade,dtype)

            try:
                return BFFileSink(header=bf_header, check_finite=output_config.check_finite)
            except BaseException:
                bf_header["fid"].close()
                raise
        
        case ".btr" | "btr":

            Nseq = 0 # will be overwritten when the file is closed

            if (pipeline_data.bands_B2 is None
                    or pipeline_data.btr_azim_deg_A is None
                    or pipeline_data.btr_elev_deg_E is None
                    or pipeline_data.btr_dtype is None):
                raise ValueError("BTR output metadata is incomplete")

            btr_header = btr_write_header(output_config.file,Nseq,pipeline_data.sample_rate_hz,
                pipeline_data.start_time_ns,pipeline_data.nfft,pipeline_data.advance,pipeline_data.bands_B2,
                pipeline_data.btr_azim_deg_A,pipeline_data.btr_elev_deg_E,pipeline_data.btr_dtype)

            try:
                return BTRFileSink(header=btr_header, check_finite=output_config.check_finite)
            except BaseException:
                btr_header["fid"].close()
                raise

        case _:
            raise ValueError(f"Unsupported output type: {output_config.type!r}")

def build_source_data_info(source,tb):

    if isinstance(source, ExistingFileSourceTS):

        # need to return the timebase stuct to pass around the pipeline data
        # need to make sure it's consistent with TS (first sample) or FFT or BF (center sample)

        return PipelineDataBlock(sample_rate_hz=source.header["sample_rate_hz"],Nchan=source.header["Nchan"],
            channels=source.header["channels"],Npt_per_seq = source.header["Npt_per_seq"],file_type=source.file_type,
            start_time_ns=int(source.header["start_unix_nanoseconds"]),timebase=tb,
            source_channel_count=source.header["Nchan"],
            source_channel_indices=np.arange(source.header["Nchan"], dtype=np.intp))

    if isinstance(source, ExistingFileSourceBTS):

        # need to return the timebase stuct to pass around the pipeline data
        # need to make sure it's consistent with TS (first sample) or FFT or BF (center sample)

        return PipelineDataBlock(sample_rate_hz=source.header["sample_rate_hz"],
            steer_rbe_rad_J3=source.header["steer_rbe_rad_J3"],Npt_per_seq = source.header["Npt_per_seq"],
            file_type=source.file_type,
            shade_N=source.header["shade"][0].copy() if source.header["Nshades"] == 1 else None,
            shade_JN=source.header["shade"].copy() if source.header["Nshades"] > 1 else None,
            start_time_ns=int(source.header["start_unix_nanoseconds"]),timebase=tb)

    elif isinstance(source, ExistingFileSourceFFT):
       
        return PipelineDataBlock(sample_rate_hz=source.header["sample_rate_hz"],Nchan=source.header["Nchan"],
            nfft=source.header["nfft"],advance=source.header["advance"],freq=source.header["freq"],
            window_T=source.header["window_T"],file_type=source.file_type,channels=source.header["channels"],
            start_time_ns=int(source.header["start_unix_nanoseconds"]),timebase=tb,
            source_channel_count=source.header["Nchan"],
            source_channel_indices=np.arange(source.header["Nchan"], dtype=np.intp))

    elif isinstance(source, ExistingFileSourceBF):
    
        return PipelineDataBlock(sample_rate_hz=source.header["sample_rate_hz"],nfft=source.header["nfft"],
            advance=source.header["advance"],freq=source.header["freq"],
            shade_N=source.header["shade"][0].copy() if source.header["Nshades"] == 1 else None,
            shade_JN=source.header["shade"].copy() if source.header["Nshades"] > 1 else None,
            steer_rbe_rad_J3=source.header["steer_rbe_rad_J3"],window_T=source.header["window_T"],
            file_type=source.file_type,start_time_ns=int(source.header["start_unix_nanoseconds"]),timebase=tb)


    
