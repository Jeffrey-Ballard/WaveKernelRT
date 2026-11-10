from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from wavekernelrt.core.bf_weight_file_utils import (bf_weight_overwrite_Nseq,bf_weight_read_header,
    bf_weight_read_multiple_sequence,bf_weight_read_single_sequence,bf_weight_write_header,bf_weight_write_sequence)


class TestBFWeightFileUtils(unittest.TestCase):
    def make_header(self,path: Path,*,Nseq=2,dtype=np.complex64):
        return bf_weight_write_header(path,Nseq=Nseq,sample_rate_hz=8.0,start_unix_nanoseconds=1_234_567_890,nfft=8,
            advance=4,freq=np.array([0.0,1.0,2.0],dtype=np.float32),
            channels=np.array([10.0,20.0],dtype=np.float32),
            steer_rbe_rad_J3=np.array([[1000.0,0.0,0.0],[1000.0,np.pi / 2,0.0],],dtype=np.float32),
            window_T=np.ones(8,dtype=np.float32),dtype=dtype)

    def test_round_trip_preserves_qjn_records_and_metadata(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weights.bfw"
            records = (np.arange(24,dtype=np.float32).reshape(2,3,2,2)
                + 1j * np.arange(100,124,dtype=np.float32).reshape(2,3,2,2)).astype(np.complex64)

            write_header = self.make_header(path)
            try:
                for heading,record_QJN in zip((10.0,20.0),records):
                    bf_weight_write_sequence(write_header,record_QJN,heading,check_finite=True)
            finally:
                write_header["fid"].close()

            header = bf_weight_read_header(path)
            try:
                self.assertEqual(header["format_version"],1)
                self.assertEqual(header["bearing_frame"],"relative")
                self.assertEqual(header["axis_order"],"QJN")
                self.assertEqual(header["weight_convention"],"conjugated_multiplication_coefficients")
                self.assertEqual(header["Nfreq"],3)
                self.assertEqual(header["Jbeams"],2)
                self.assertEqual(header["Nchan"],2)
                self.assertEqual(header["dtype"],np.dtype(np.complex64))

                record_QJN,heading = bf_weight_read_single_sequence(header,1)
                np.testing.assert_array_equal(record_QJN,records[1])
                self.assertEqual(heading,20.0)

                records_QJNT,headings_T = bf_weight_read_multiple_sequence(header,[1,0,1])
                self.assertEqual(records_QJNT.shape,(3,2,2,3))
                np.testing.assert_array_equal(records_QJNT,np.moveaxis(records[[1,0,1]],0,-1))
                np.testing.assert_array_equal(headings_T,[20.0,10.0,20.0])
            finally:
                header["fid"].close()

    def test_records_are_multiplication_coefficients(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weights.bfw"
            mathematical_w_QJN = np.array(
                [
                    [[[1 + 2j,3 - 4j],[2 - 1j,-1 + 3j]]],
                    [[[4 + 1j,2 + 2j],[3 - 2j,1 - 1j]]],
                    [[[2 + 0j,1 + 5j],[4 + 3j,2 - 2j]]],
                ],
                dtype=np.complex64,
            ).reshape(3,2,2)
            stored_QJN = np.conj(mathematical_w_QJN)
            data_QN = np.array([[1 + 1j,2 - 1j],[3 + 2j,1 + 0j],[2 - 3j,4 + 1j]],dtype=np.complex64)

            write_header = self.make_header(path,Nseq=1)
            try:
                bf_weight_write_sequence(write_header,stored_QJN,45.0)
            finally:
                write_header["fid"].close()

            header = bf_weight_read_header(path)
            try:
                loaded_QJN,heading = bf_weight_read_single_sequence(header,0)
            finally:
                header["fid"].close()

            expected_QJ = np.einsum("qn,qjn->qj",data_QN,np.conj(mathematical_w_QJN))
            actual_QJ = np.einsum("qn,qjn->qj",data_QN,loaded_QJN)
            np.testing.assert_allclose(actual_QJ,expected_QJ)
            self.assertEqual(heading,45.0)

    def test_overwrite_count_and_shape_validation(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weights.bfw"
            header = self.make_header(path,Nseq=0)
            try:
                with self.assertRaisesRegex(ValueError,"must have shape"):
                    bf_weight_write_sequence(header,np.zeros((3,2),dtype=np.complex64),0.0)
                bf_weight_write_sequence(header,np.zeros((3,2,2),dtype=np.complex64),370.0)
                bf_weight_overwrite_Nseq(header,1)
            finally:
                header["fid"].close()

            read_header = bf_weight_read_header(path)
            try:
                self.assertEqual(read_header["Nseq"],1)
                record,heading = bf_weight_read_single_sequence(read_header,0)
                np.testing.assert_array_equal(record,np.zeros((3,2,2),dtype=np.complex64))
                self.assertEqual(heading,10.0)
            finally:
                read_header["fid"].close()

    def test_header_requires_complex_dtype(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weights.bfw"
            with self.assertRaisesRegex(ValueError,"complex dtype"):
                self.make_header(path,dtype=np.float32)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
