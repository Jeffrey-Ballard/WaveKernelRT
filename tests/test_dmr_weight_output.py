from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from wavekernelrt.core.bf_file_utils import bf_read_header,bf_read_multiple_sequence
from wavekernelrt.core.bf_weight_file_utils import bf_weight_read_header,bf_weight_read_multiple_sequence
from wavekernelrt.core.wng_file_utils import wng_read_header,wng_read_multiple_sequence
from wavekernelrt.core.fft_file_utils import fft_write_header,fft_write_sequence
from wavekernelrt.core.geometry import Geometry
from wavekernelrt.pipeline.build_pipeline import _resolve_dmr_weight_output_file,_resolve_dmr_wng_output_file,build_pipeline
from wavekernelrt.pipeline.configs import DMRElementConfig,InputConfig,OutputConfig,PipelineConfig,ShadingConfig
from wavekernelrt.pipeline.run_pipeline import run_pipeline


class TestDMRWeightOutput(unittest.TestCase):
    def write_fft(self,path: Path,records_TQN):
        header = fft_write_header(path,Nseq=len(records_TQN),sample_rate_hz=800.0,start_unix_nanoseconds=2_000_000_000,
            nfft=8,advance=4,freq=np.array([100.0,200.0,300.0],dtype=np.float32),
            channels=np.array([0.0,1.0],dtype=np.float32),window_T=np.ones(8,dtype=np.float32),dtype=np.complex64)
        try:
            for record_QN in records_TQN:
                fft_write_sequence(header,record_QN)
        finally:
            header["fid"].close()

    def make_config(self,geometry,input_file,beam_file,weight_file,*,weight_band_Hz=(150.0,300.0),
        weight_seq_stride=1,wng_seq_stride=1,heading_provider=None):
        dmr = DMRElementConfig(geom=geometry,shading=ShadingConfig(window_type="uniform"),
            steer_rbe_rad_J3=np.array([[1_000_000.0,0.0,0.0]],dtype=np.float32),Dmax=1,Dmin=1,wngc_val_dB=-6.0,
            wngc_tol_dB=1.0,eig_thresh=0.9,write_weights=True,weight_band_Hz=weight_band_Hz,
            output_weight_file=weight_file,weight_seq_stride=weight_seq_stride,write_wng=True,
            wng_band_Hz=weight_band_Hz,wng_seq_stride=wng_seq_stride,heading_provider=heading_provider)
        return PipelineConfig(input=InputConfig(file=input_file,type="fft"),stages=(dmr,),output=OutputConfig(file=beam_file,type="bfft"))

    def test_band_limited_weights_reproduce_dmr_stage_output(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            geometry_file = root / "line.txt"
            geometry_file.write_text("0 0 0\n0 20 0\n")
            geometry = Geometry(geometry_file,array_type="line")

            input_file = root / "input.fft"
            beam_file = root / "output.bfft"
            weight_file = beam_file.with_suffix(".bfw")
            wng_file = beam_file.with_suffix(".wng")
            base_records_TQN = np.array([[[1 + 2j,3 - 1j],[2 - 1j,-1 + 0.5j],[4 + 0j,2 + 3j]],
                                         [[2 + 1j,1 - 2j],[3 + 0.5j,2 - 1j],[-1 + 2j,4 + 0j]]],
                                        dtype=np.complex64)
            # Dmax=1 and the default 1 dB loss target train for five records,
            # so the sixth record exercises adaptive DMR-weight storage.
            records_TQN = np.concatenate((base_records_TQN,1.5 * base_records_TQN,0.75 * base_records_TQN),
                axis=0).astype(np.complex64)
            self.write_fft(input_file,records_TQN)

            pipeline = build_pipeline(
                self.make_config(geometry,input_file,beam_file,None,weight_seq_stride=5,wng_seq_stride=5,
                    heading_provider=lambda block: 350.0 + 7.0 * block.sequence))
            run_pipeline(pipeline.source,pipeline.stages,pipeline.sink)

            weight_header = bf_weight_read_header(weight_file)
            try:
                self.assertEqual(weight_header["Nseq"],2)
                self.assertEqual(weight_header["advance"],20)
                self.assertAlmostEqual(weight_header["secs_per_seq"],0.025)
                np.testing.assert_array_equal(weight_header["freq"],np.array([200.0,300.0],dtype=np.float32))
                weights_QJNT,weight_headings_T = bf_weight_read_multiple_sequence(weight_header,range(2))
            finally:
                weight_header["fid"].close()

            wng_header = wng_read_header(wng_file)
            try:
                self.assertEqual(wng_header["Nseq"],2)
                self.assertEqual(wng_header["advance"],20)
                self.assertAlmostEqual(wng_header["secs_per_seq"],0.025)
                self.assertEqual(wng_header["dtype"],np.dtype(np.float32))
                self.assertEqual(wng_header["wngc_val_dB"],-6.0)
                self.assertEqual(wng_header["wngc_tol_dB"],1.0)
                np.testing.assert_array_equal(wng_header["freq"],np.array([200.0,300.0],dtype=np.float32))
                wng_QJT,wng_headings_T = wng_read_multiple_sequence(wng_header,range(2))
            finally:
                wng_header["fid"].close()

            with self.assertRaisesRegex(ValueError,"BF header"):
                bf_read_header(wng_file)

            beam_header = bf_read_header(beam_file)
            try:
                beamformed_QJT,beam_headings_T = bf_read_multiple_sequence(beam_header,range(6))
            finally:
                beam_header["fid"].close()

            for stored_sequence,source_sequence in enumerate((0,5)):
                self.assertEqual(weight_headings_T[stored_sequence],beam_headings_T[source_sequence])
                self.assertEqual(wng_headings_T[stored_sequence],beam_headings_T[source_sequence])
                self.assertEqual(beam_headings_T[source_sequence],(350.0 + 7.0 * source_sequence) % 360.0)
                reconstructed_QJ = np.einsum("qn,qjn->qj",records_TQN[source_sequence,1:3],weights_QJNT[:,:,:,stored_sequence])
                np.testing.assert_allclose(reconstructed_QJ,beamformed_QJT[1:3,:,source_sequence],rtol=1e-6,atol=1e-6)
                expected_wng_QJ = 1.0 / np.sum(np.abs(weights_QJNT[:,:,:,stored_sequence]) ** 2,axis=2)
                np.testing.assert_allclose(wng_QJT[:,:,stored_sequence],expected_wng_QJ,rtol=2e-5,atol=2e-5)

    def test_weight_output_defaults_off(self):
        config = DMRElementConfig(geom=None,shading=ShadingConfig(window_type="uniform"),
            steer_rbe_rad_J3=[[1_000_000.0,0.0,0.0]],Dmax=1,Dmin=1,wngc_val_dB=-6.0,wngc_tol_dB=1.0,eig_thresh=0.9)
        self.assertFalse(config.write_weights)
        self.assertIsNone(config.weight_band_Hz)
        self.assertIsNone(config.output_weight_file)
        self.assertEqual(config.weight_seq_stride,1)
        self.assertFalse(config.write_wng)
        self.assertIsNone(config.wng_band_Hz)
        self.assertIsNone(config.output_wng_file)
        self.assertEqual(config.wng_seq_stride,1)

    def test_wng_output_is_independent_of_weight_output(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            geometry_file = root / "line.txt"
            geometry_file.write_text("0 0 0\n0 20 0\n")
            geometry = Geometry(geometry_file,array_type="line")
            input_file = root / "input.fft"
            beam_file = root / "output.bfft"
            self.write_fft(input_file,np.ones((21,3,2),dtype=np.complex64))

            dmr = DMRElementConfig(geom=geometry,shading=ShadingConfig(window_type="uniform"),
                steer_rbe_rad_J3=np.array([[1_000_000.0,0.0,0.0]],dtype=np.float32),Dmax=1,Dmin=1,wngc_val_dB=-6.0,
                wngc_tol_dB=1.0,eig_thresh=0.9,write_wng=True,wng_band_Hz=(200.0,200.0),wng_seq_stride=10)
            pipeline = build_pipeline(PipelineConfig(
                input=InputConfig(file=input_file,type="fft"),
                stages=(dmr,),
                output=OutputConfig(file=beam_file,type="bfft"),
            ))
            run_pipeline(pipeline.source,pipeline.stages,pipeline.sink)

            self.assertFalse(beam_file.with_suffix(".bfw").exists())
            wng_file = beam_file.with_suffix(".wng")
            header = wng_read_header(wng_file)
            try:
                self.assertEqual(header["Nseq"],3)
                self.assertEqual(header["advance"],40)
                self.assertAlmostEqual(header["secs_per_seq"],0.05)
                self.assertEqual(header["wngc_val_dB"],-6.0)
                self.assertEqual(header["wngc_tol_dB"],1.0)
                np.testing.assert_array_equal(header["freq"],[200.0])
                expected_times_ns = np.array([
                    header["start_unix_nanoseconds"]
                    + round(
                        index
                        * header["advance"]
                        * 1_000_000_000
                        / float(header["sample_rate_hz"]))
                    for index in range(header["Nseq"])
                ])
                np.testing.assert_array_equal(expected_times_ns,np.array([2_000_000_000,2_050_000_000,2_100_000_000]))
                wng_QJT,headings_T = wng_read_multiple_sequence(header,range(3))
                wng_QJ = wng_QJT[:,:,0]
                self.assertEqual(headings_T.shape,(3,))
                self.assertTrue(np.all(np.isfinite(wng_QJ)))
                self.assertTrue(np.all(wng_QJ > 0))
            finally:
                header["fid"].close()

    def test_enabled_weight_output_allows_derived_filename(self):
        config = DMRElementConfig(geom=None,shading=ShadingConfig(window_type="uniform"),
            steer_rbe_rad_J3=[[1_000_000.0,0.0,0.0]],Dmax=1,Dmin=1,wngc_val_dB=-6.0,wngc_tol_dB=1.0,eig_thresh=0.9,
            write_weights=True)
        self.assertIsNone(config.output_weight_file)

    def test_weight_filename_resolution_priority(self):
        config = DMRElementConfig(geom=None,shading=ShadingConfig(window_type="uniform"),
            steer_rbe_rad_J3=[[1_000_000.0,0.0,0.0]],Dmax=1,Dmin=1,wngc_val_dB=-6.0,wngc_tol_dB=1.0,eig_thresh=0.9,
            write_weights=True)
        input_config = InputConfig(file=Path("/data/event.input.fft"),type="fft")

        bpsd_output = OutputConfig(file=Path("/results/beams.bpsd"),type="bpsd")
        self.assertEqual(_resolve_dmr_weight_output_file(config,input_config,bpsd_output),Path("/results/beams.bfw"))

        btr_output = OutputConfig(file=Path("/results/detections.btr"),type="btr")
        self.assertEqual(_resolve_dmr_weight_output_file(config,input_config,btr_output),Path("/results/event.input_dmr.bfw"))

        explicit_config = DMRElementConfig(geom=None,shading=ShadingConfig(window_type="uniform"),
            steer_rbe_rad_J3=[[1_000_000.0,0.0,0.0]],Dmax=1,Dmin=1,wngc_val_dB=-6.0,wngc_tol_dB=1.0,eig_thresh=0.9,
            write_weights=True,output_weight_file=Path("/custom/weights.bfw"))
        self.assertEqual(_resolve_dmr_weight_output_file(explicit_config,input_config,bpsd_output),Path("/custom/weights.bfw"))

        self.assertEqual(_resolve_dmr_wng_output_file(config,input_config,bpsd_output),Path("/results/beams.wng"))
        self.assertEqual(_resolve_dmr_wng_output_file(config,input_config,btr_output),Path("/results/event.input_dmr.wng"))

        explicit_wng_config = DMRElementConfig(geom=None,shading=ShadingConfig(window_type="uniform"),
            steer_rbe_rad_J3=[[1_000_000.0,0.0,0.0]],Dmax=1,Dmin=1,wngc_val_dB=-6.0,wngc_tol_dB=1.0,eig_thresh=0.9,
            write_wng=True,output_wng_file=Path("/custom/wng.wng"))
        self.assertEqual(_resolve_dmr_wng_output_file(explicit_wng_config,input_config,bpsd_output),Path("/custom/wng.wng"))


if __name__ == "__main__":
    unittest.main()
