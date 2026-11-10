from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from wavekernelrt.core.bts_file_utils import bts_read_header,bts_read_sequence,bts_write_header,bts_write_sequence


class TestBTSHeading(unittest.TestCase):
    def test_round_trip_returns_one_heading_per_record(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "beams.bts"
            steer_J3 = np.array([
                [1000.0,0.0,0.0],
                [1000.0,np.pi / 2,0.0],
            ],dtype=np.float32)
            header = bts_write_header(path,Nseq=2,sample_rate_hz=8.0,start_unix_nanoseconds=1_000_000_000,
                Npt_per_seq=3,shade=np.ones(2,dtype=np.float32),steer_rbe_rad_J3=steer_J3)
            records = np.arange(12,dtype=np.float32).reshape(2,2,3)
            try:
                bts_write_sequence(header,records[0],359.0)
                bts_write_sequence(header,records[1],361.0)
            finally:
                header["fid"].close()

            header = bts_read_header(path)
            try:
                self.assertEqual(header["format_version"],1)
                self.assertEqual(header["bearing_frame"],"relative")
                data_JT,headings_T = bts_read_sequence(header,[1,0])
            finally:
                header["fid"].close()

            np.testing.assert_array_equal(data_JT,np.concatenate((records[1],records[0]),axis=1))
            np.testing.assert_array_equal(headings_T,[1.0,359.0])


if __name__ == "__main__":
    unittest.main()
