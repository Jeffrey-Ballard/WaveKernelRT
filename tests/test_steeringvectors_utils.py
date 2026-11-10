import unittest

import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from wavekernelrt.core.geometry import Geometry
from wavekernelrt.core.steeringvectors import SteeringVectors
from wavekernelrt.core.steeringvectors_utils import construct_rbe_X3,convert_rbe_to_xyz


class TestSteeringVectorUtils(unittest.TestCase):

    def test_replica_matches_numpy_fft_phase_for_delayed_signal(self):
        with TemporaryDirectory() as temp_dir:
            geometry_file = Path(temp_dir) / "geometry.txt"
            geometry_file.write_text(
                "0 0 0\n"
                "40 80 0\n"
                "100 150 0\n"
                "180 210 0\n")
            geometry = Geometry(geometry_file,array_type="line",soundspeed_ftpers=5000)
            frequency_hz = np.array([500.0])
            source_bearing_deg = 70.0
            source = construct_rbe_X3(1_000_000,source_bearing_deg,0.0)
            steering = SteeringVectors(geometry,source,frequency_hz,flag_relative=False)

            expected_QN = np.exp(-1j * 2 * np.pi * frequency_hz[:,None] * steering.time_delays_NJ[:,0][None,:])
            np.testing.assert_allclose(steering.complex_phasors_QNJ[:,:,0],expected_QN,rtol=1e-6,atol=1e-6)

    def test_construct_rbe_uses_expected_order_and_units(self):
        actual = construct_rbe_X3(np.array([3.2808,6.5616]),np.array([0.0,90.0]),np.array([-10.0,10.0]))

        self.assertEqual(actual.shape,(8,3))
        np.testing.assert_allclose(actual[:,0],[1,1,1,1,2,2,2,2])
        np.testing.assert_allclose(actual[:,1],np.deg2rad([0,0,90,90,0,0,90,90]),rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(actual[:,2],np.deg2rad([-10,10,-10,10,-10,10,-10,10]),rtol=1e-6,atol=1e-7)

    def test_convert_rbe_to_xyz_uses_navigation_bearings(self):
        rbe_rad_K3 = np.array([[2.0,0.0,0.0],[2.0,np.pi / 2,0.0],[2.0,0.0,np.pi / 2],])

        actual = convert_rbe_to_xyz(rbe_rad_K3)

        np.testing.assert_allclose(actual,[[0,2,0],[2,0,0],[0,0,2]],atol=1e-6)


if __name__ == "__main__":
    unittest.main()
