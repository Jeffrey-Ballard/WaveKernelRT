from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.signal.windows import hann

from wavekernelrt.core.fft_file_utils import fft_read_header,fft_read_single_sequence
from wavekernelrt.core.ts_file_utils import ts_write_header,ts_write_sequence
from wavekernelrt.pipeline.build_pipeline import build_pipeline
from wavekernelrt.pipeline.configs import FFTtoPSDConfig,InputConfig,OutputConfig,PipelineConfig,TStoFFTConfig
from wavekernelrt.pipeline.run_pipeline import run_pipeline
from wavekernelrt.processing.TStoFFT_Stage import TStoFFT_Stage


class TestFFTWindowNormalization(unittest.TestCase):
    def test_stage_power_normalizes_supplied_window(self):
        nfft = 64
        stage = TStoFFT_Stage(nfft=nfft,advance=nfft // 2,sample_rate_hz=4096.0,file_type="fft",window_T=hann(nfft,sym=True))

        self.assertEqual(stage.window_T.dtype,np.float32)
        self.assertAlmostEqual(float(np.mean(stage.window_T**2)),1.0,places=6)

    def test_psd_header_stores_applied_window_and_recovers_hann_tone(self):
        sample_rate_hz = 4096.0
        nfft = 4096
        frequency_hz = 256.0
        expected_amplitude = 2.0

        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            ts_file = temp_root / "sinusoid.ts"
            psd_file = temp_root / "sinusoid.psd"

            time_secs = np.arange(nfft) / sample_rate_hz
            sinusoid = expected_amplitude * np.sin(2 * np.pi * frequency_hz * time_secs)

            ts_header = ts_write_header(ts_file,Nseq=1,sample_rate_hz=sample_rate_hz,start_unix_nanoseconds=0,
                Npt_per_seq=nfft,channels=np.array([0.0]))
            try:
                ts_write_sequence(ts_header,sinusoid[np.newaxis,:])
            finally:
                ts_header["fid"].close()

            pipeline = build_pipeline(PipelineConfig(
                input=InputConfig(file=ts_file,type="ts"),
                stages=(TStoFFTConfig(nfft=nfft,advance=nfft,window_name="hann"),FFTtoPSDConfig()),
                output=OutputConfig(file=psd_file,type="psd"),
            ))
            applied_window = pipeline.stages[0].window_T.copy()
            run_pipeline(pipeline.source,pipeline.stages,pipeline.sink)

            psd_header = fft_read_header(psd_file)
            try:
                stored_window = psd_header["window_T"]
                psd_QN = fft_read_single_sequence(psd_header,0)
            finally:
                psd_header["fid"].close()

        np.testing.assert_array_equal(stored_window,applied_window)
        self.assertAlmostEqual(float(np.mean(stored_window**2)),1.0,places=6)

        df_hz = sample_rate_hz / nfft
        tone_bin = round(frequency_hz / df_hz)
        window_mean = np.mean(stored_window)
        enbw_bins = np.mean(stored_window**2) / window_mean**2
        recovered_amplitude = np.sqrt(2 * psd_QN[tone_bin,0] * df_hz * enbw_bins)

        self.assertAlmostEqual(float(recovered_amplitude),expected_amplitude,places=5)


if __name__ == "__main__":
    unittest.main()
