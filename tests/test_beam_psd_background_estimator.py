import unittest

import numpy as np
from scipy.ndimage import convolve1d

from wavekernelrt.processing.BeamPSDBackgroundEstimator import BeamPSDBackgroundEstimator
from wavekernelrt.pipeline.configs import BackgroundEstimateConfig


class TestBeamPSDBackgroundEstimator(unittest.TestCase):
    def make_estimator(self,*,time_records=None,sector_beams=1,full_circle=False,
        smoothing_window=np.array([1.0]),car_parms=(),nbeams=4):
        return BeamPSDBackgroundEstimator(bkgd_time_seq_avg=time_records,nbeams=nbeams,
            bkgd_sector_beam_size=sector_beams,full_circle_flag=full_circle,freq_smoothing_window=smoothing_window,
            car_parms_WP=car_parms)

    def test_waits_for_full_window_and_returns_center_record(self):
        estimator = self.make_estimator(time_records=3,nbeams=2)

        self.assertIsNone(estimator.process(np.full((2,2),1.0)))
        self.assertIsNone(estimator.process(np.full((2,2),4.0)))

        estimate = estimator.process(np.full((2,2),7.0))
        self.assertIsNotNone(estimate)
        np.testing.assert_array_equal(estimate.center_psd_QJ,np.full((2,2),4.0))
        np.testing.assert_allclose(estimate.background_dB_QJ,10 * np.log10(4.0))

        estimate = estimator.process(np.full((2,2),10.0))
        np.testing.assert_array_equal(estimate.center_psd_QJ,np.full((2,2),7.0))
        np.testing.assert_allclose(estimate.background_dB_QJ,10 * np.log10(7.0))

    def test_bounded_sector_uses_only_available_edge_beams(self):
        estimator = self.make_estimator(sector_beams=3)
        estimate = estimator.process(np.array([[1.0,100.0,10.0,1000.0]]))

        expected_linear = np.array([[100.0,10.0,100.0,1000.0]])
        np.testing.assert_allclose(estimate.background_dB_QJ,10 * np.log10(expected_linear))

    def test_full_circle_sector_wraps_at_edges(self):
        estimator = self.make_estimator(sector_beams=3,full_circle=True)
        estimate = estimator.process(np.array([[1.0,100.0,10.0,1000.0]]))

        expected_linear = np.array([[100.0,10.0,100.0,10.0]])
        np.testing.assert_allclose(estimate.background_dB_QJ,10 * np.log10(expected_linear))

    def test_rejects_unexpected_beam_count(self):
        estimator = self.make_estimator()
        with self.assertRaises(ValueError):
            estimator.process(np.ones((2,3)))

    def test_time_average_can_run_without_other_filters(self):
        estimator = self.make_estimator(time_records=3,sector_beams=None,smoothing_window=None,car_parms=None,nbeams=2)

        self.assertIsNone(estimator.process(np.full((2,2),1.0)))
        self.assertIsNone(estimator.process(np.full((2,2),4.0)))
        estimate = estimator.process(np.full((2,2),7.0))

        np.testing.assert_allclose(estimate.background_dB_QJ,10 * np.log10(4.0))

    def test_sector_width_is_validated_during_construction(self):
        with self.assertRaises(ValueError):
            self.make_estimator(sector_beams=5,nbeams=4)

    def test_frequency_smoothing_can_run_without_time_average(self):
        smoothing_window = np.array([0.25,0.5,0.25])
        estimator = self.make_estimator(time_records=None,sector_beams=None,smoothing_window=smoothing_window,car_parms=None,nbeams=2)
        data_QJ = np.array([
            [1.0,10.0],
            [100.0,1000.0],
            [10.0,100.0],
        ])

        estimate = estimator.process(data_QJ)

        expected_dB_QJ = convolve1d(10 * np.log10(data_QJ),smoothing_window,axis=0,mode="reflect",origin=0)
        np.testing.assert_array_equal(estimate.center_psd_QJ,data_QJ)
        np.testing.assert_allclose(estimate.background_dB_QJ,expected_dB_QJ)

    def test_background_config_defaults_to_all_steps_disabled(self):
        config = BackgroundEstimateConfig()

        self.assertIsNone(config.bkgd_time_seq_avg)
        self.assertIsNone(config.bkgd_sector_beam_size)
        self.assertIsNone(config.freq_smoothing_window)
        self.assertIsNone(config.car_parms_WP)
        self.assertFalse(config.full_circle_flag)


if __name__ == "__main__":
    unittest.main()
