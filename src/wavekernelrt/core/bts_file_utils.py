"""Binary I/O for beam time-series (BTS) files."""

from os import fstat

import numpy as np

from wavekernelrt.core.validate_utils import validate_header_count, validate_header_positive_float32, validate_header_timestamp
from wavekernelrt.core.validate_utils import (validate_header_shading,validate_header_beam_grid,validate_read_indices,
                                              validate_array_heading_deg)

_MAGIC = 0x12345678
_FORMAT_VERSION = 1
_FIXED_HEADER_BYTES = 4 + 4 + 6 * 4 + 8
_DTYPE = np.dtype(np.float32)

def bts_write_header(filename,Nseq,sample_rate_hz,start_unix_nanoseconds,Npt_per_seq,shade,steer_rbe_rad_J3):

    # Make sure inputs are what WaveKernelRT will expect downstream
    Nseq = validate_header_count(Nseq, "Nseq", minimum=0)
    Npt_per_seq = validate_header_count(Npt_per_seq, "Npt_per_seq")

    sample_rate_hz = validate_header_positive_float32(sample_rate_hz, "sample_rate_hz")

    start_unix_nanoseconds = validate_header_timestamp(start_unix_nanoseconds)

    steer_rbe_rad_J3 = validate_header_beam_grid(steer_rbe_rad_J3)

    shade = validate_header_shading(shade, steer_rbe_rad_J3.shape[0])

    # make sure to close it out if it fails on construction
    fid = open(filename,"wb")
    try:

        header = dict()
        header["fid"] = fid
        header["format_version"] = _FORMAT_VERSION
        header["bearing_frame"] = "relative"
        header["sample_rate_hz"] = sample_rate_hz
        header["Nseq"] = int(Nseq)
        header["Nchan"] = int(shade.shape[1])
        header["Nshades"] = int(shade.shape[0])
        header["shade"] = shade
        header["Jbeams"] = int(steer_rbe_rad_J3.shape[0])
        header["steer_rbe_rad_J3"] = np.asarray(steer_rbe_rad_J3).astype(np.float32)
        header["Npt_per_seq"] = int(Npt_per_seq)

        header["start_unix_seconds"] = start_unix_nanoseconds/1e9
        header["start_unix_nanoseconds"] = int(start_unix_nanoseconds) # to be consistent with reading header

        header["byte_header"] = _FIXED_HEADER_BYTES + _DTYPE.itemsize * (header["Nshades"] * header["Nchan"] + header["Jbeams"] * 3)

        np.array([_MAGIC],dtype=np.uint32).tofile(fid)
        np.array([_FORMAT_VERSION],dtype=np.uint32).tofile(fid)
        np.array([Nseq],dtype=np.float32).tofile(fid)
        np.array([sample_rate_hz],dtype=np.float32).tofile(fid)
        np.array([Npt_per_seq],dtype=np.float32).tofile(fid)
        np.array(header["Jbeams"],dtype=np.float32).tofile(fid)
        np.array(header["Nchan"], dtype=np.float32).tofile(fid)
        np.array(header["Nshades"], dtype=np.float32).tofile(fid)
        np.array(header["steer_rbe_rad_J3"],dtype=np.float32).tofile(fid)
        header["shade"].tofile(fid)
        np.array([start_unix_nanoseconds], dtype=np.uint64).tofile(fid)

        return header
    except BaseException:
        fid.close()
        raise

def bts_read_header(filename):

    fid = open(filename,"rb")
    try:
        endian_test = fid.read(4)

        if len(endian_test) != 4:
            raise ValueError("Truncated BTS header: missing byte-order marker")

        endian_big = np.frombuffer(endian_test, dtype=">u4")[0]
        endian_little = np.frombuffer(endian_test, dtype="<u4")[0]

        if endian_big == _MAGIC:
            endian = ">"
        elif endian_little == _MAGIC:
            endian = "<"
        else:
            raise ValueError("BTS byte-order marker not recognized")

        version = np.fromfile(fid, dtype=endian + "u4", count=1)
        if version.size != 1 or int(version.item()) != _FORMAT_VERSION:
            raise ValueError(f"Unsupported BTS format version: {version.item() if version.size else 'missing'}")

        tmp = np.fromfile(fid, dtype=endian+"f4", count=6)
        if tmp.size != 6:
            raise ValueError("Incomplete BTS header")
        
        header = {"fid": fid,"format_version": _FORMAT_VERSION,"bearing_frame": "relative"}

        for name, value, minimum in (("Nseq", tmp[0], 0),("Npt_per_seq", tmp[2], 1),("Jbeams", tmp[3], 1),
            ("Nchan", tmp[4], 1),("Nshades", tmp[5], 1)):
            if not np.isfinite(value) or value != np.floor(value):
                raise ValueError(
                    f"BTS header {name} must be a finite integer; "
                    f"received {value!r}")

            header[name] = validate_header_count(int(value), name, minimum=minimum)

        header["sample_rate_hz"] = validate_header_positive_float32(tmp[1], "sample_rate_hz")

        if header["Nshades"] not in (1, header["Jbeams"]):
            raise ValueError("BTS Nshades must be 1 or Jbeams")

        ncoordinates = header["Jbeams"] * 3
        nweights = header["Nshades"] * header["Nchan"]

        # Bound variable-length reads using the actual file size.
        header["byte_header"] = _FIXED_HEADER_BYTES + _DTYPE.itemsize * (ncoordinates + nweights)
        if fstat(fid.fileno()).st_size < header["byte_header"]:
            raise ValueError("Truncated BTS header: dimensions exceed available bytes")

        steer = np.fromfile(fid, dtype=endian + "f4", count=ncoordinates)
        if steer.size != ncoordinates:
            raise ValueError("Incomplete BTS steering grid")

        header["steer_rbe_rad_J3"] = validate_header_beam_grid(steer.reshape(header["Jbeams"], 3))

        shade = np.fromfile(fid, dtype=endian + "f4", count=nweights)
        if shade.size != nweights:
            raise ValueError("Incomplete BTS shading data")

        header["shade"] = validate_header_shading(shade.reshape(header["Nshades"], header["Nchan"]),header["Jbeams"])

        timestamp = np.fromfile(fid, dtype=endian + "u8", count=1)
        if timestamp.size != 1:
            raise ValueError("Truncated BTS header: missing timestamp")

        # dividing by 1e9 loses precision in the nanoseconds for a float, store the nanosecond directly
        header["start_unix_nanoseconds"] = validate_header_timestamp(timestamp.item())
        header["start_unix_seconds"] = header["start_unix_nanoseconds"] / 1e9

        header["secs_per_record"] = header["Npt_per_seq"] / header["sample_rate_hz"]
        header["total_time_secs"] = header["Nseq"] * header["secs_per_record"]
        header["endian"] = endian

        bytes_per_record = _DTYPE.itemsize * (1 + header["Jbeams"] * header["Npt_per_seq"])
        expected_file_bytes = header["byte_header"] + header["Nseq"] * bytes_per_record

        actual_file_bytes = fstat(fid.fileno()).st_size

        if actual_file_bytes < expected_file_bytes:
            raise ValueError(
                f"Truncated BTS data: header declares {header['Nseq']} records, "
                f"requiring {expected_file_bytes} bytes; "
                f"file contains {actual_file_bytes} bytes")

        return header
    except BaseException:
        fid.close()
        raise

def bts_write_sequence(header,data_JT,array_heading_deg, *, check_finite=False):

    data_JT = np.asarray(data_JT)

    expected_shape = (header["Jbeams"], header["Npt_per_seq"])
    if data_JT.shape != expected_shape:
        raise ValueError(
            f"BTS record must have shape {expected_shape}; "
            f"received {data_JT.shape}")

    if data_JT.dtype.kind not in "iuf":
        raise ValueError("BTS samples must be real numeric values")

    if check_finite and not np.all(np.isfinite(data_JT)):
        raise ValueError("BTS samples must be finite")

    data_float32 = data_JT.astype(_DTYPE, copy=False)

    if check_finite and not np.all(np.isfinite(data_float32)):
        raise ValueError("BTS samples must remain finite after float32 conversion")

    heading = validate_array_heading_deg(array_heading_deg)
    np.asarray(heading, dtype=np.float32).tofile(header["fid"])
    data_float32.tofile(header["fid"])

def bts_read_sequence(header,seqs_to_read):
    """Return concatenated beam samples and one heading per selected record."""

    # data is laid out as (they are all float32)
    # [header ][ch 0 seq 0][ch 1 seq 0][... ch N-1 seq 0][ch 0 seq 1][ch 0 seq 1][ch 1 seq 1][... ch N-1 seq 1]
    seqs_to_read = validate_read_indices(seqs_to_read, header["Nseq"], "seqs_to_read")

    fid = header["fid"]
    Jbeams = header["Jbeams"]
    npts = len(seqs_to_read)*header["Npt_per_seq"]

    data_JT = np.empty((Jbeams,npts),dtype=_DTYPE)
    headings_T = np.empty(len(seqs_to_read), dtype=_DTYPE)
    data_bytes_per_record = header["Npt_per_seq"] * Jbeams * _DTYPE.itemsize
    bytes_per_record = _DTYPE.itemsize + data_bytes_per_record
    for idx,seq in enumerate(seqs_to_read):
        # move the pointer to the right location to read
        seq = int(seq) # force it to python int which can grow without wrapping
        byte_start_seq = header["byte_header"] + seq*bytes_per_record
        fid.seek(byte_start_seq,0)

        heading = np.fromfile(fid, dtype=header["endian"] + "f4", count=1)
        if heading.size != 1:
            raise ValueError(f"Truncated BTS heading at sequence {seq}")
        headings_T[idx] = validate_array_heading_deg(heading.item())
        
        temp = np.fromfile(fid,dtype=header["endian"]+"f4",count=header["Npt_per_seq"]*Jbeams)
        idx1 = idx*header["Npt_per_seq"]
        idx2 = idx1 + header["Npt_per_seq"]
        data_JT[:,idx1:idx2] = temp.reshape((Jbeams,header["Npt_per_seq"]))

    return data_JT, headings_T

def bts_overwrite_Nseq(header,Nseq):
    Nseq = validate_header_count(Nseq, "Nseq", minimum=0)
    header["fid"].seek(4+4,0)
    np.asarray(Nseq, dtype=np.float32).tofile(header["fid"])
