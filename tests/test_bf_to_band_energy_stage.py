import unittest

import numpy as np

from wavekernelrt.core.steeringvectors_utils import construct_rbe_X3
from wavekernelrt.pipeline.blocks import BFBlock
from wavekernelrt.processing.BeamPSDBackgroundEstimator import BeamPSDBackgroundEstimator
from wavekernelrt.processing.BFtoBandEnergy_Stage import BFtoBandEnergy_Stage


class TestBFtoBandEnergyStage(unittest.TestCase):
    def make_grid(self,azimuths=(0.0,90.0),elevations=(0.0,)):
        return construct_rbe_X3(np.array([1000.0]),np.asarray(azimuths),np.asarray(elevations))

    def make_block(self,data_QJ,sequence=0):
        return BFBlock(data_QJ=np.asarray(data_QJ),advance=1,sequence=sequence,array_heading_deg=25.0 + sequence)

    def test_direct_integration_includes_frequency_bin_width(self):
        grid = self.make_grid(azimuths=(0.0,90.0),elevations=(-10.0,10.0))
        stage = BFtoBandEnergy_Stage(freq=np.array([0.0,2.0,4.0,6.0]),bands_B2=[(0.0,2.0),(4.0,6.0)],steer_rbe_rad_J3=grid)
        data_QJ = np.array([
            [1.0,2.0,3.0,4.0],
            [10.0,20.0,30.0,40.0],
            [100.0,200.0,300.0,400.0],
            [1000.0,2000.0,3000.0,4000.0],
        ])

        output = list(stage.process(self.make_block(data_QJ)))[0]

        expected_linear_BAE = np.array([
            [[22.0,44.0],[66.0,88.0]],
            [[2200.0,4400.0],[6600.0,8800.0]],
        ])
        expected_BAE = (10 * np.log10(expected_linear_BAE)).astype(np.float32)
        np.testing.assert_array_equal(output.data_BAE,expected_BAE)
        self.assertEqual(output.data_BAE.dtype,np.float32)

    def test_background_normalization_uses_centered_record(self):
        grid = self.make_grid()
        estimator = BeamPSDBackgroundEstimator(nbeams=2,bkgd_time_seq_avg=3)
        stage = BFtoBandEnergy_Stage(freq=np.array([0.0,2.0,4.0]),bands_B2=[(0.0,4.0)],steer_rbe_rad_J3=grid,
            background_estimator=estimator)

        self.assertEqual(list(stage.process(self.make_block(np.full((3,2),1.0)))),[])
        self.assertEqual(list(stage.process(self.make_block(np.full((3,2),2.0)))),[])
        output = list(stage.process(self.make_block(np.full((3,2),9.0))))[0]

        # Center PSD is 2 and background mean is 4. Three bins at df=2.
        np.testing.assert_allclose(output.data_BAE,10 * np.log10(3.0))

    def test_float32_output_floors_zero_at_minus_400_db(self):
        stage = BFtoBandEnergy_Stage(freq=np.array([0.0,1.0]),bands_B2=[(0.0,1.0)],steer_rbe_rad_J3=self.make_grid())

        output = list(stage.process(self.make_block(np.zeros((2,2)))))[0]

        np.testing.assert_array_equal(output.data_BAE,np.full((1,2,1),-400.0,dtype=np.float32))

    def test_rank_output_is_independent_for_each_elevation(self):
        grid = self.make_grid(azimuths=(0.0,90.0,180.0),elevations=(-10.0,10.0))
        stage = BFtoBandEnergy_Stage(freq=np.array([0.0,1.0]),bands_B2=[(0.0,1.0)],steer_rbe_rad_J3=grid,output="rank_uint8")
        desired_AE = np.array([
            [1.0,30.0],
            [3.0,20.0],
            [2.0,10.0],
        ])
        data_QJ = np.vstack((desired_AE.reshape(-1) / 2,) * 2)

        output = list(stage.process(self.make_block(data_QJ)))[0]

        expected_BAE = np.array([[
            [0,255],
            [255,128],
            [128,0],
        ]],dtype=np.uint8)
        np.testing.assert_array_equal(output.data_BAE,expected_BAE)

    def test_nonuniform_frequency_grid_uses_per_bin_widths(self):
        stage = BFtoBandEnergy_Stage(freq=np.array([0.0,1.0,3.0]),bands_B2=[(0.0,3.0)],steer_rbe_rad_J3=self.make_grid())

        output = list(stage.process(self.make_block(np.ones((3,2)))))[0]

        # Bin widths are [1.0, 1.5, 2.0] from one-sided endpoint
        # differences and centered interior differences.
        np.testing.assert_allclose(output.data_BAE,10 * np.log10(4.5))

    def test_rejects_band_without_frequency_bins(self):
        with self.assertRaisesRegex(ValueError,"at least one bin"):
            BFtoBandEnergy_Stage(freq=np.array([0.0,1.0,2.0]),bands_B2=[(3.0,4.0)],steer_rbe_rad_J3=self.make_grid())

    def test_rejects_multiple_ranges(self):
        grid = construct_rbe_X3(np.array([1000.0,2000.0]),np.array([0.0,90.0]),np.array([0.0]))
        with self.assertRaisesRegex(NotImplementedError,"one steering range"):
            BFtoBandEnergy_Stage(freq=np.array([0.0,1.0]),bands_B2=[(0.0,1.0)],steer_rbe_rad_J3=grid)


if __name__ == "__main__":
    unittest.main()
