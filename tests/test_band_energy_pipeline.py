from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from wavekernelrt.core.bf_file_utils import bf_write_header,bf_write_sequence
from wavekernelrt.core.btr_file_utils import btr_read_header,btr_read_sequence
from wavekernelrt.core.steeringvectors_utils import construct_rbe_X3
from wavekernelrt.pipeline.build_pipeline import build_pipeline
from wavekernelrt.pipeline.configs import BackgroundEstimateConfig,BandEnergyConfig,InputConfig,OutputConfig,PipelineConfig
from wavekernelrt.pipeline.run_pipeline import run_pipeline


class TestBandEnergyPipeline(unittest.TestCase):
    sample_rate_hz = 8.0
    nfft = 8
    advance = 8
    start_time_ns = 1_000_000_000

    def write_bpsd(self,path: Path,records_QJ):
        freq = np.array([0.0,1.0,2.0],dtype=np.float32)
        steer_rbe_rad_J3 = construct_rbe_X3(np.array([1000.0]),np.array([0.0,90.0]),np.array([0.0]))
        header = bf_write_header(path,Nseq=len(records_QJ),sample_rate_hz=self.sample_rate_hz,
            start_unix_nanoseconds=self.start_time_ns,nfft=self.nfft,advance=self.advance,freq=freq,
            steer_rbe_rad_J3=steer_rbe_rad_J3,window_T=np.ones(self.nfft,dtype=np.float32),
            shade=np.ones(2,dtype=np.float32),dtype=np.float32)
        try:
            for index,record_QJ in enumerate(records_QJ):
                bf_write_sequence(header,np.asarray(record_QJ,dtype=np.float32),100.0 + index)
        finally:
            header["fid"].close()

    def run_band_energy(self,input_file,output_file,stage_config):
        pipeline = build_pipeline(PipelineConfig(
            input=InputConfig(file=input_file,type="bpsd"),
            stages=(stage_config,),
            output=OutputConfig(file=output_file,type="btr"),
        ))
        run_pipeline(pipeline.source,pipeline.stages,pipeline.sink)

    def test_float32_output_header_and_integrated_values(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_file = root / "input.bpsd"
            output_file = root / "output.btr"
            records = [
                np.full((3,2),1.0),
                np.full((3,2),2.0),
            ]
            self.write_bpsd(input_file,records)

            self.run_band_energy(input_file,output_file,BandEnergyConfig(bands_B2=[(0.0,2.0)],output="float32"))

            header = btr_read_header(output_file)
            try:
                data_EBAT,headings_T = btr_read_sequence(header,range(header["Nseq"]))
                self.assertEqual(header["dtype"],np.dtype(np.float32))
                self.assertEqual(header["Nseq"],2)
                self.assertEqual(header["start_unix_nanoseconds"],self.start_time_ns)
                np.testing.assert_allclose(data_EBAT[0,0],10 * np.log10(np.array([[3.0,6.0],[3.0,6.0]])))
                np.testing.assert_array_equal(headings_T,[100.0,101.0])
            finally:
                header["fid"].close()

    def test_optional_time_background_updates_timestamp_and_record_count(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_file = root / "input.bpsd"
            output_file = root / "output.btr"
            records = [
                np.full((3,2),1.0),
                np.full((3,2),2.0),
                np.full((3,2),9.0),
            ]
            self.write_bpsd(input_file,records)

            self.run_band_energy(input_file,output_file,
                BandEnergyConfig(bands_B2=[(0.0,2.0)],background=BackgroundEstimateConfig(bkgd_time_seq_avg=3),output="float32"))

            header = btr_read_header(output_file)
            try:
                data_EBAT,headings_T = btr_read_sequence(header,0)
                self.assertEqual(header["Nseq"],1)
                self.assertEqual(header["start_unix_nanoseconds"],self.start_time_ns + 1_000_000_000)
                np.testing.assert_allclose(data_EBAT,10 * np.log10(1.5),rtol=1e-6)
                np.testing.assert_array_equal(headings_T,[101.0])
            finally:
                header["fid"].close()

    def test_rank_output_selects_uint8_btr_storage(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_file = root / "input.bpsd"
            output_file = root / "output.btr"
            self.write_bpsd(input_file,[np.array([[1.0,3.0],[1.0,3.0],[1.0,3.0],])])

            self.run_band_energy(input_file,output_file,BandEnergyConfig(bands_B2=[(0.0,2.0)],output="rank_uint8"))

            header = btr_read_header(output_file)
            try:
                data_EBAT,headings_T = btr_read_sequence(header,0)
                self.assertEqual(header["dtype"],np.dtype(np.uint8))
                np.testing.assert_array_equal(data_EBAT[:,:,:,0],np.array([[[0,255]]],dtype=np.uint8))
                np.testing.assert_array_equal(headings_T,[100.0])
            finally:
                header["fid"].close()


if __name__ == "__main__":
    unittest.main()
