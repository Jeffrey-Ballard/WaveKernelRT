from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from wavekernelrt.core import bf_file_utils,bf_weight_file_utils,btr_file_utils,bts_file_utils,fft_file_utils,ts_file_utils,wng_file_utils


class TestFileFormatVersions(unittest.TestCase):
    def test_every_public_binary_format_starts_at_version_one(self):
        modules = (ts_file_utils,bts_file_utils,fft_file_utils,bf_file_utils,wng_file_utils,bf_weight_file_utils,btr_file_utils)
        self.assertTrue(all(module._FORMAT_VERSION == 1 for module in modules))

    def test_ts_header_writes_reads_and_validates_version_one(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "versioned.ts"
            header = ts_file_utils.ts_write_header(path,Nseq=0,sample_rate_hz=8_000.0,start_unix_nanoseconds=0,
                Npt_per_seq=8,channels=[0,1])
            ts_file_utils.ts_write_sequence(header,np.zeros((2,8)))
            ts_file_utils.ts_overwrite_Nseq(header,1)
            header["fid"].close()

            with path.open("rb") as stream:
                stream.seek(4)
                self.assertEqual(np.fromfile(stream,dtype=np.uint32,count=1)[0],1)

            header = ts_file_utils.ts_read_header(path)
            try:
                self.assertEqual(header["format_version"],1)
                self.assertEqual(header["byte_header"],40)
                self.assertEqual(header["Nseq"],1)
            finally:
                header["fid"].close()

            self._replace_version(path,2)
            with self.assertRaisesRegex(ValueError,"Unsupported TS format version: 2"):
                ts_file_utils.ts_read_header(path)

    def test_fft_header_writes_reads_and_validates_version_one(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "versioned.fft"
            nfft = 8
            sample_rate_hz = 8_000.0
            header = fft_file_utils.fft_write_header(path,Nseq=0,sample_rate_hz=sample_rate_hz,
                start_unix_nanoseconds=0,nfft=nfft,advance=4,freq=np.fft.rfftfreq(nfft,d=1 / sample_rate_hz),
                channels=[0,1],window_T=np.ones(nfft),dtype=np.complex64)
            fft_file_utils.fft_write_sequence(header,np.zeros((5,2)))
            fft_file_utils.fft_overwrite_Nseq(header,1)
            header["fid"].close()

            with path.open("rb") as stream:
                stream.seek(4)
                self.assertEqual(np.fromfile(stream,dtype=np.uint32,count=1)[0],1)

            header = fft_file_utils.fft_read_header(path)
            try:
                self.assertEqual(header["format_version"],1)
                expected_size = 49 + 4 * (nfft + 5 + 2)
                self.assertEqual(header["byte_header"],expected_size)
                self.assertEqual(header["Nseq"],1)
            finally:
                header["fid"].close()

            self._replace_version(path,2)
            with self.assertRaisesRegex(ValueError,"Unsupported FFT format version: 2"):
                fft_file_utils.fft_read_header(path)

    @staticmethod
    def _replace_version(path: Path,version: int) -> None:
        with path.open("r+b") as stream:
            stream.seek(4)
            np.asarray(version,dtype=np.uint32).tofile(stream)


if __name__ == "__main__":
    unittest.main()
