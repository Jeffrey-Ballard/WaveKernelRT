import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wavekernelrt.core.analysis_images import btr_image


class TestBTRImage(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_colorbar_is_disabled_by_default(self):
        fig,ax,mesh = btr_image(np.ones((2,3)))

        self.assertEqual(fig.axes,[ax])
        self.assertIsNone(mesh.colorbar)

    def test_optional_colorbar_preserves_data_axis_width(self):
        data = np.arange(6,dtype=np.float32).reshape(2,3)
        fig_without,ax_without,_ = btr_image(data,pixels_per_sample=2.0)
        data_width_without = (ax_without.get_position().width * fig_without.get_figwidth())

        fig_with,ax_with,mesh = btr_image(data,pixels_per_sample=2.0,colorbar=True)
        data_width_with = ax_with.get_position().width * fig_with.get_figwidth()

        self.assertEqual(len(fig_with.axes),2)
        self.assertIsNotNone(mesh.colorbar)
        self.assertAlmostEqual(data_width_with,data_width_without)
        np.testing.assert_allclose(fig_without.get_size_inches(),fig_with.get_size_inches())
        np.testing.assert_allclose(ax_without.get_position().bounds,ax_with.get_position().bounds)

    def test_colorbar_must_be_boolean(self):
        with self.assertRaisesRegex(ValueError,"colorbar"):
            btr_image(np.ones((2,3)),colorbar="yes")

    def test_bearing_ticks_adapt_to_initial_data_area_width(self):
        bearings = np.linspace(0.0,360.0,200,endpoint=False)
        _,ax,_ = btr_image(np.ones((2,bearings.size)),col_data=bearings,pixels_per_sample=1.0)

        np.testing.assert_allclose(np.diff(ax.get_xticks()),90.0)


if __name__ == "__main__":
    unittest.main()
