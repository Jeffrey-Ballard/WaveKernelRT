import unittest
from types import SimpleNamespace

import numpy as np

from wavekernelrt.core.general_utils import set_bearing_ticks_for_width


class _StubAxis:
    def __init__(self,width_px,device_pixel_ratio):
        self._width_px = width_px
        self.figure = SimpleNamespace(canvas=SimpleNamespace(device_pixel_ratio=device_pixel_ratio))
        self.ticks = None

    def get_xlim(self):
        return 0.0,360.0

    def get_window_extent(self):
        return SimpleNamespace(width=self._width_px)

    def set_xticks(self,ticks):
        self.ticks = ticks


class TestSetBearingTicksForWidth(unittest.TestCase):
    def test_device_pixel_ratio_does_not_change_logical_tick_spacing(self):
        standard_dpi_axis = _StubAxis(width_px=360.0,device_pixel_ratio=1.0)
        high_dpi_axis = _StubAxis(width_px=720.0,device_pixel_ratio=2.0)

        standard_interval = set_bearing_ticks_for_width(standard_dpi_axis)
        high_dpi_interval = set_bearing_ticks_for_width(high_dpi_axis)

        self.assertEqual(standard_interval,45.0)
        self.assertEqual(high_dpi_interval,standard_interval)
        np.testing.assert_array_equal(high_dpi_axis.ticks,standard_dpi_axis.ticks)


if __name__ == "__main__":
    unittest.main()
