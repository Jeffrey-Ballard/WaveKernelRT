import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wavekernelrt.core.analysis_images import wng_image_QJ,wng_line_T
from wavekernelrt.core.wng_file_utils import wng_constraint_linear_J


class TestWNGPlots(unittest.TestCase):
    def setUp(self):
        self.wng_QJT = np.array([[[1.0,2.0],[2.0,4.0],[4.0,8.0]],[[1.5,3.0],[3.0,6.0],[6.0,12.0]],],dtype=np.float32)

    def tearDown(self):
        plt.close("all")

    def test_plain_image_displays_selected_record_in_db(self):
        fig,ax,mesh = wng_image_QJ(self.wng_QJT,idxT=1,freq=[100.0,200.0],beams=[0.0,30.0,60.0])

        expected_QJ = 10.0 * np.log10(self.wng_QJT[:,:,1])
        np.testing.assert_allclose(np.asarray(mesh.get_array()),expected_QJ)
        self.assertIsNotNone(mesh.colorbar)
        self.assertEqual(mesh.get_cmap().name,"viridis")
        self.assertEqual(ax.get_xlabel(),"Relative bearing (deg)")
        self.assertIn("idxT=1",ax.get_title())
        self.assertEqual(fig.axes[0],ax)

    def test_constraint_image_shows_clipped_signed_db_difference(self):
        constraint_J = np.array([2.0,5.0,7.0])
        _,ax,mesh = wng_image_QJ(self.wng_QJT,idxT=1,constraint_wng=constraint_J)

        expected_QJ = np.clip(10.0 * np.log10(constraint_J) - 10.0 * np.log10(self.wng_QJT[:,:,1]),-10.0,10.0)
        np.testing.assert_allclose(np.asarray(mesh.get_array()),expected_QJ)
        self.assertEqual(mesh.get_clim(),(-10.0,10.0))
        self.assertIn("Effective WNGC:",ax.get_title())
        self.assertIn("negative passes, positive fails",ax.get_title())

        # The passing side ends at white and the failing side immediately
        # jumps to Viridis so crossing the constraint is conspicuous.
        midpoint_colors = mesh.get_cmap()(np.array([127 / 255,128 / 255]))
        np.testing.assert_allclose(midpoint_colors[0,:3],1.0)
        np.testing.assert_allclose(midpoint_colors[1],plt.get_cmap("jet")(0.0))
        np.testing.assert_allclose(mesh.get_cmap()(0.0)[:3],0.0)
        self.assertFalse(np.allclose(mesh.get_cmap()(1.0)[:3],1.0))

    def test_absolute_and_constraint_images_share_plot_layout(self):
        plain_fig,plain_ax,plain_mesh = wng_image_QJ(self.wng_QJT,idxT=1,title="First title line\nSecond title line")
        constraint_fig,constraint_ax,constraint_mesh = wng_image_QJ(self.wng_QJT,idxT=1,
            constraint_wng=np.array([2.0,5.0,7.0]),title="First title line\nSecond title line")

        plain_fig.canvas.draw()
        constraint_fig.canvas.draw()

        np.testing.assert_allclose(plain_ax.get_position().bounds,constraint_ax.get_position().bounds)
        np.testing.assert_allclose(plain_mesh.colorbar.ax.get_position().bounds,constraint_mesh.colorbar.ax.get_position().bounds)
        self.assertLessEqual(constraint_ax.title.get_window_extent().y1,constraint_fig.bbox.y1)

    def test_line_plots_selected_trace_and_constraint(self):
        times = np.array([0.0,0.5])
        _,ax,line = wng_line_T(self.wng_QJT,idxQ=1,idxJ=2,times=times,freq=[100.0,200.0],beams=[0.0,30.0,60.0],
            constraint_wng=[2.0,4.0,8.0])

        np.testing.assert_array_equal(line.get_xdata(),times)
        np.testing.assert_allclose(line.get_ydata(),10.0 * np.log10(self.wng_QJT[1,2,:]))
        np.testing.assert_allclose(ax.lines[1].get_ydata(),10.0 * np.log10(8.0))
        self.assertIn("200 Hz",ax.get_title())
        self.assertIn("60 deg",ax.get_title())

    def test_constraint_uses_active_aperture_and_tolerance(self):
        header = {
            "shade": np.array([[1.0,0.0,0.5],[1.0,1.0,1.0]]),
            "Jbeams": 2,
            "wngc_val_dB": -3.0,
            "wngc_tol_dB": 1.0,
        }

        np.testing.assert_allclose(wng_constraint_linear_J(header),np.array([2.0,3.0]) * 10.0 ** (-4.0 / 10.0))

    def test_rejects_nonpositive_wng(self):
        invalid = self.wng_QJT.copy()
        invalid[0,0,0] = 0.0
        with self.assertRaisesRegex(ValueError,"strictly positive"):
            wng_image_QJ(invalid)


if __name__ == "__main__":
    unittest.main()
