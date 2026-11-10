from pathlib import Path
import unittest

import numpy as np
from scipy.signal.windows import hamming

from wavekernelrt.core.beampattern import BeamPattern
from wavekernelrt.core.geometry import Geometry
from wavekernelrt.core.shading import Shading
from wavekernelrt.pipeline.configs import ShadingConfig


GEOMETRY_ROOT = Path(__file__).resolve().parents[1] / "examples" / "basic_geometries_and_beampatterns"


class TestHammingShading(unittest.TestCase):
    def test_config_accepts_hamming(self):
        config = ShadingConfig(window_type="hamming")
        self.assertEqual(config.window_type,"hamming")

    def test_line_array_matches_symmetric_hamming_window(self):
        geometry = Geometry(GEOMETRY_ROOT / "simple_line_array.txt",array_type="line")
        actual = np.asarray(Shading(geometry).hamming_shading_N).reshape(-1)

        expected = hamming(actual.size,sym=True)
        expected /= expected.sum()

        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-8)
        self.assertEqual(actual.dtype,np.float32)

    def test_panel_array_uses_separable_hamming_window(self):
        geometry = Geometry(GEOMETRY_ROOT / "simple_panel_array.txt",array_type="panel_starboard")
        actual = np.asarray(Shading(geometry).hamming_shading_N).reshape(geometry.Ncols,geometry.Nrows)

        expected = hamming(geometry.Ncols,sym=True)[:,None] * hamming(geometry.Nrows,sym=True)[None,:]
        expected /= expected.sum()

        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-8)

    def test_curved_array_builds_normalized_per_beam_windows(self):
        geometry = Geometry(GEOMETRY_ROOT / "simple_cylinder_array.txt",array_type="cylinder")
        shading = Shading(geometry)
        beams = np.array([
            [1_000_000.0,0.0,0.0],
            [1_000_000.0,np.pi / 2,0.0],
        ])

        actual = shading.hamming_shading_for_beams(beams)

        self.assertEqual(actual.shape,(2,geometry.xyz_m_N3.shape[0]))
        np.testing.assert_allclose(actual.sum(axis=1),1.0,rtol=1e-6)

        for weights,look in zip(actual,beams):
            visible = shading.visible_elements(look)
            self.assertTrue(np.all(weights[visible] > 0))
            self.assertTrue(np.all(weights[~visible] == 0))

    def test_beampattern_accepts_hamming_config(self):
        geometry = Geometry(GEOMETRY_ROOT / "simple_line_array.txt",array_type="line")
        beampattern = BeamPattern(geometry,freq=1_000,Nbearings=5,Nelevations=1)
        beampattern.set_shading_config(ShadingConfig(window_type="hamming"))

        weights = beampattern.cbf_weight_QNJ

        self.assertEqual(weights.shape,(1,geometry.xyz_m_N3.shape[0],5))
        np.testing.assert_allclose(np.abs(weights).sum(axis=1),1.0,rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
