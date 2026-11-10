import unittest
from math import ceil

import numpy as np

from wavekernelrt.core.geometry import BeamformingGeometry
from wavekernelrt.pipeline.blocks import FFTBlock
from wavekernelrt.processing.FFTtoDMRElement_Stage import FFTtoDMRElement_Stage


class TestDMRTrainingPolicy(unittest.TestCase):
    def setUp(self):
        geometry = BeamformingGeometry(array_type="line",xyz_m_N3=np.array([[0.0,-0.5,0.0],[0.0,0.5,0.0]]),
            phase_center_m_13=np.zeros((1,3)),normals_N3=None,soundspeed_mpers=np.array(1500.0),array_heading_deg=0.0,
            far_field_range_m=1000.0,max_aperture_m=1.0)
        self.stage_kwargs = dict(geom=geometry,shade=np.ones(2),steer_rbe_rad_J3=np.array([[1_000_000.0,0.0,0.0]]),
            freq_process=np.array([100.0,200.0]),Dmax=10,Dmin=1,wngc_val_dB=-6.0,wngc_tol_dB=1.0,eig_thresh=0.9)
        self.stage = FFTtoDMRElement_Stage(**self.stage_kwargs)

    def test_reference_count_and_ema_have_consistent_effective_count(self):
        expected_training = ceil(self.stage.Dmax / (1.0 - 10.0**(-self.stage.sinr_loss_dB / 10.0)))

        self.assertEqual(self.stage.training_snapshots,expected_training)
        self.assertEqual(self.stage.sinr_loss_dB,1.0)
        self.assertEqual(expected_training,49)
        self.assertAlmostEqual(self.stage.alpha,2.0 / 50.0)
        self.assertAlmostEqual((2.0 - self.stage.alpha) / self.stage.alpha,expected_training)

    def test_configured_loss_changes_the_snapshot_count(self):
        stage = FFTtoDMRElement_Stage(**self.stage_kwargs,sinr_loss_dB=3.0)
        expected_training = ceil(10 / (1.0 - 10.0**(-3.0 / 10.0)))

        self.assertEqual(stage.training_snapshots,expected_training)
        self.assertEqual(expected_training,21)

    def test_loss_must_be_positive(self):
        with self.assertRaisesRegex(ValueError,"sinr_loss_dB"):
            FFTtoDMRElement_Stage(**self.stage_kwargs,sinr_loss_dB=0.0)

    def test_startup_accumulates_an_equal_weight_sample_covariance(self):
        data_QN = np.array([[1.0 + 2.0j,3.0 - 1.0j],[2.0 - 0.5j,-1.0 + 1.0j]],dtype=np.complex64)
        block = FFTBlock(data_QN=data_QN,advance=1,sequence=0)

        for _ in range(self.stage.training_snapshots):
            list(self.stage.process(block))

        expected_QNN = np.einsum("qn,qm->qnm",data_QN,data_QN.conj())
        np.testing.assert_allclose(self.stage.raw_csm_qNN,expected_QNN,rtol=2e-6,atol=2e-6)
        self.assertEqual(self.stage.count,self.stage.training_snapshots)


if __name__ == "__main__":
    unittest.main()
