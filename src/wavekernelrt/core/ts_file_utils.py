"""Binary I/O for channel time-series (TS) files."""

import sys
from os import fstat

import numpy as np

from wavekernelrt.core.validate_utils import validate_header_count, validate_header_positive_float32, validate_header_array
from wavekernelrt.core.validate_utils import validate_header_timestamp, validate_read_indices

_MAGIC = 0x12345678
_FORMAT_VERSION = 1
_FIXED_HEADER_BYTES = 4 + 4 + 4 * 4 + 8
_DTYPE = np.dtype(np.float32)

def ts_write_header(filename,Nseq,sample_rate_hz,start_unix_nanoseconds,Npt_per_seq,channels):

    """Write a TS header, validating all inputs and enforcing a data contract
    before opening the file.

    Counts must be integers exactly representable in the format's float32
    fields. Channels must be a nonempty, finite real vector. Sample rate
    and channels are serialized as float32; the returned header uses those
    same values. The Unix timestamp is an exact uint64 nanosecond count.
    """

    channels = validate_header_array(channels, "channels", shape=(None,))

    Nseq = validate_header_count(Nseq, "Nseq", minimum=0)
    Npt_per_seq = validate_header_count(Npt_per_seq, "Npt_per_seq")
    Nchan = validate_header_count(channels.size, "Nchan")

    sample_rate_hz = validate_header_positive_float32(sample_rate_hz, "sample_rate_hz")

    start_unix_nanoseconds = validate_header_timestamp(start_unix_nanoseconds)

    # make sure to close it out if it fails on construction
    fid = open(filename,"wb")
    try:
        header = dict()
        header["fid"] = fid
        header["format_version"] = _FORMAT_VERSION
        header["Nseq"] = int(Nseq)
        header["sample_rate_hz"] = sample_rate_hz
        header["Nchan"] = Nchan
        header["channels"] = channels
        header["Npt_per_seq"] = int(Npt_per_seq)
        header["start_unix_seconds"] = start_unix_nanoseconds / 1e9
        header["start_unix_nanoseconds"] = int(start_unix_nanoseconds) # to be consistent with reading header
        header["endianess"] = sys.byteorder
        header["secs_per_record"] = header["Npt_per_seq"] / float(header["sample_rate_hz"])
        header["total_time_secs"] = header["Nseq"] * header["secs_per_record"]
        header["byte_header"] = _FIXED_HEADER_BYTES + header["Nchan"] * _DTYPE.itemsize

        np.array([_MAGIC],dtype=np.uint32).tofile(fid)
        np.array([_FORMAT_VERSION],dtype=np.uint32).tofile(fid)
        np.array([Nseq],dtype=np.float32).tofile(fid)
        np.array([sample_rate_hz],dtype=np.float32).tofile(fid)
        np.array([Npt_per_seq],dtype=np.float32).tofile(fid)
        np.array([Nchan],dtype=np.float32).tofile(fid)
        np.array([channels],dtype=np.float32).tofile(fid)
        np.array([start_unix_nanoseconds], dtype=np.uint64).tofile(fid)

        return header
    except BaseException:
        fid.close()
        raise


def ts_read_header(filename):

    fid = open(filename,"rb")
    
    try:
        endian_test = fid.read(4)

        if len(endian_test) != 4:
            raise ValueError("Truncated TS header: missing byte-order marker")

        endian_big = np.frombuffer(endian_test, dtype=">u4")[0]
        endian_little = np.frombuffer(endian_test, dtype="<u4")[0]

        if endian_big == _MAGIC:
            endian = ">"
        elif endian_little == _MAGIC:
            endian = "<"
        else:
            raise ValueError("TS byte-order marker not recognized")

        version = np.fromfile(fid, dtype=endian + "u4", count=1)
        if version.size != 1 or int(version.item()) != _FORMAT_VERSION:
            received = version.item() if version.size else "missing"
            raise ValueError(f"Unsupported TS format version: {received}")

        tmp = np.fromfile(fid, dtype=endian + "f4", count=4)
        if tmp.size != 4:
            raise ValueError("Truncated TS header: expected four metadata values")

        header = {"fid": fid, "format_version": _FORMAT_VERSION}

        for name, value, minimum in (("Nseq", tmp[0], 0),("Npt_per_seq", tmp[2], 1),("Nchan", tmp[3], 1)):
            if not np.isfinite(value) or value != np.floor(value):
                raise ValueError(f"TS header {name} must be a finite integer; received {value!r}")

            header[name] = validate_header_count(int(value), name, minimum=minimum)

        header["sample_rate_hz"] = validate_header_positive_float32(tmp[1], "sample_rate_hz")

        # Ensure the declared channel vector and timestamp fit before reading them.
        header["byte_header"] = _FIXED_HEADER_BYTES + header["Nchan"] * _DTYPE.itemsize
        if fstat(fid.fileno()).st_size < header["byte_header"]:
            raise ValueError("Truncated TS header: channel count exceeds available bytes")

        channels = np.fromfile(fid, dtype=endian + "f4", count=header["Nchan"])

        header["channels"] = validate_header_array(channels, "channels", shape=(header["Nchan"],))

        timestamp = np.fromfile(fid, dtype=endian + "u8", count=1)
        if timestamp.size != 1:
            raise ValueError("Truncated TS header: missing timestamp")

        # dividing by 1e9 loses precision in the nanoseconds for a float, store the nanosecond directly
        header["start_unix_nanoseconds"] = validate_header_timestamp(timestamp.item())
        header["start_unix_seconds"] = header["start_unix_nanoseconds"] / 1e9

        header["secs_per_record"] = header["Npt_per_seq"] / header["sample_rate_hz"]
        header["total_time_secs"] = header["Nseq"] * header["secs_per_record"]
        header["endian"] = endian

        bytes_per_record = header["Nchan"] * header["Npt_per_seq"] * _DTYPE.itemsize
        expected_file_bytes = header["byte_header"] + header["Nseq"] * bytes_per_record
        
        actual_file_bytes = fstat(fid.fileno()).st_size

        if actual_file_bytes < expected_file_bytes:
            raise ValueError(
                f"Truncated TS data: header declares {header['Nseq']} records, "
                f"requiring {expected_file_bytes} bytes; "
                f"file contains {actual_file_bytes} bytes")

        return header
    except BaseException:
        fid.close()
        raise

def ts_write_sequence(header,data_NT, *, check_finite=False):

    data_NT = np.asarray(data_NT)

    expected_shape = (header["Nchan"], header["Npt_per_seq"])
    if data_NT.shape != expected_shape:
        raise ValueError(
            f"TS record must have shape {expected_shape}; "
            f"received {data_NT.shape}")
    
    if data_NT.dtype.kind not in "iuf":
        raise ValueError("TS samples must be real numeric values")

    if check_finite and not np.all(np.isfinite(data_NT)):
        raise ValueError("TS samples must be finite")

    data_float32 = data_NT.astype(_DTYPE, copy=False)

    if check_finite and not np.all(np.isfinite(data_float32)):
        raise ValueError("TS samples must remain finite after float32 conversion")

    data_float32.tofile(header["fid"])
    
def ts_read_sequence(header,seqs_to_read):

    # data is laid out as (they are all float32)
    # [header ][ch 0 seq 0][ch 1 seq 0][... ch N-1 seq 0][ch 0 seq 1][ch 0 seq 1][ch 1 seq 1][... ch N-1 seq 1]
    seqs_to_read = validate_read_indices(seqs_to_read, header["Nseq"], "seqs_to_read")

    fid = header["fid"]
    nchan = header["Nchan"]
    npts = len(seqs_to_read)*header["Npt_per_seq"]

    data_NT = np.empty((nchan,npts),dtype=_DTYPE)
    for idx,seq in enumerate(seqs_to_read):
        # move the pointer to the right location to read
        seq = int(seq) # force it to python int which can grow without wrapping
        byte_start_seq = header["byte_header"] + seq * header["Npt_per_seq"] * nchan * _DTYPE.itemsize
        fid.seek(byte_start_seq,0)
        
        temp = np.fromfile(fid,dtype=header["endian"]+"f4",count=header["Npt_per_seq"]*nchan)
        idx1 = idx*header["Npt_per_seq"]
        idx2 = idx1 + header["Npt_per_seq"]
        data_NT[:,idx1:idx2] = temp.reshape((nchan,header["Npt_per_seq"]))

    return data_NT

def ts_overwrite_Nseq(header,Nseq):
    Nseq = validate_header_count(Nseq, "Nseq", minimum=0)
    header["fid"].seek(4+4,0)
    np.asarray(Nseq, dtype=np.float32).tofile(header["fid"])
