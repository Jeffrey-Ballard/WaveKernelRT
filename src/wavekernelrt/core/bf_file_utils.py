"""Binary I/O for beam frequency-domain (BFFT/BPSD) files."""

from os import fstat

import numpy as np

from wavekernelrt.core.data_types import dtype_to_code, code_to_dtype, dtype_to_bytes
from wavekernelrt.core.validate_utils import validate_header_count, validate_header_positive_float32, validate_header_timestamp
from wavekernelrt.core.validate_utils import (validate_header_array,validate_header_frequencies,
    validate_header_beam_grid,validate_header_shading,validate_read_indices,validate_array_heading_deg)

_MAGIC = 0x12345678
_FORMAT_VERSION = 1
_FIXED_HEADER_BYTES = 4 + 4 + 9 * 4 + 8 + 8 + 1

def bf_write_header(filename,Nseq,sample_rate_hz,start_unix_nanoseconds,nfft,advance,freq,steer_rbe_rad_J3,window_T,shade,dtype):

    Nseq = validate_header_count(Nseq, "Nseq", minimum=0)
    nfft = validate_header_count(nfft, "nfft")
    advance = validate_header_count(advance, "advance")

    sample_rate_hz = validate_header_positive_float32(sample_rate_hz, "sample_rate_hz")
    start_unix_nanoseconds = validate_header_timestamp(start_unix_nanoseconds)

    freq = validate_header_frequencies(freq, sample_rate_hz, nfft)
    window_T = validate_header_array(window_T, "window_T", shape=(nfft,))
    steer_rbe_rad_J3 = validate_header_beam_grid(steer_rbe_rad_J3)

    Jbeams = steer_rbe_rad_J3.shape[0]
    shade = validate_header_shading(shade, Jbeams)

    Nfreq = validate_header_count(freq.size, "Nfreq")
    Nshades, Nchan = shade.shape

    # Sequence duration is stored as float32; header size as uint64.
    secs_per_seq = validate_header_positive_float32(advance / float(sample_rate_hz), "secs_per_seq")

    byte_header = _FIXED_HEADER_BYTES + 4 * (nfft + Nshades * Nchan + Nfreq + Jbeams * 3)

    dtype = np.dtype(dtype)
    dtype_code = dtype_to_code(dtype)

    # make sure to close it out if it fails on construction
    fid = open(filename,"wb")
    try:
        
        header = dict()
        header["fid"] = fid
        header["format_version"] = _FORMAT_VERSION
        header["bearing_frame"] = "relative"
        header["sample_rate_hz"] = sample_rate_hz
        header["Nseq"] = int(Nseq)
        header["Jbeams"] = int(steer_rbe_rad_J3.shape[0])
        header["nfft"] = int(nfft)
        header["window_T"] = np.asarray(window_T).astype(np.float32) 
        header["Nfreq"] = len(freq)
        header["advance"] = int(advance)
        header["Nshades"] = int(shade.shape[0])
        header["Nchan"] = int(shade.shape[1])
        header["shade"] = shade
        header["freq"] = np.asarray(freq).astype(np.float32)
        header["steer_rbe_rad_J3"] = np.asarray(steer_rbe_rad_J3).astype(np.float32)
        header["dtype"] = dtype
        header["start_unix_seconds"] = start_unix_nanoseconds/1e9
        header["start_unix_nanoseconds"] = int(start_unix_nanoseconds) # to be consistent with reading header
                
        header["secs_per_seq"] = secs_per_seq
        header["byte_header"] = byte_header

        np.array([_MAGIC],dtype=np.uint32).tofile(fid)
        np.array([_FORMAT_VERSION],dtype=np.uint32).tofile(fid)
        np.array([sample_rate_hz],dtype=np.float32).tofile(fid)
        np.array([Nseq],dtype=np.float32).tofile(fid)
        np.array(header["Jbeams"],dtype=np.float32).tofile(fid)
        np.array([nfft],dtype=np.float32).tofile(fid)
        np.array(header["Nfreq"],dtype=np.float32).tofile(fid)
        np.array([advance],dtype=np.float32).tofile(fid)
        np.array(header["Nchan"], dtype=np.float32).tofile(fid)
        np.array(header["Nshades"], dtype=np.float32).tofile(fid)        
        np.array(header["secs_per_seq"],dtype=np.float32).tofile(fid)
        np.array([header["byte_header"]], dtype=np.uint64).tofile(fid)
        np.array([start_unix_nanoseconds], dtype=np.uint64).tofile(fid)
        np.array([dtype_code],dtype=np.uint8).tofile(fid)
        np.array([window_T],dtype=np.float32).tofile(fid)
        header["shade"].tofile(fid)
        np.array([freq],dtype=np.float32).tofile(fid)
        np.array(header["steer_rbe_rad_J3"],dtype=np.float32).tofile(fid)
    
        return header
    except BaseException:
        fid.close()
        raise

def bf_write_sequence(header, PSD_QJ, array_heading_deg, *, check_finite=False):

    data_QJ = np.asarray(PSD_QJ)

    expected_shape = (header["Nfreq"], header["Jbeams"])
    if data_QJ.shape != expected_shape:
        raise ValueError(
            f"BF record must have shape {expected_shape}; "
            f"received {data_QJ.shape}")

    output_dtype = np.dtype(header["dtype"])

    if data_QJ.dtype.kind not in "iufc":
        raise ValueError("BF record must contain real or complex numeric values")

    if not np.can_cast(data_QJ.dtype, output_dtype, casting="same_kind"):
        raise ValueError(
            f"Cannot convert BF record from {data_QJ.dtype} to {output_dtype} "
            "using same-kind casting")

    if check_finite and not np.all(np.isfinite(data_QJ)):
        raise ValueError("BF record values must be finite")

    data_out = data_QJ.astype(output_dtype, copy=False)

    if check_finite and not np.all(np.isfinite(data_out)):
        raise ValueError("BF record values must remain finite after conversion")

    heading = validate_array_heading_deg(array_heading_deg)
    np.asarray(heading, dtype=np.float32).tofile(header["fid"])
    data_out.tofile(header["fid"])

def bf_read_header(filename):
    
    fid = open(filename,"rb")

    try:
        endian_test = fid.read(4)

        if len(endian_test) != 4:
            raise ValueError("Truncated BF header: missing byte-order marker")

        endian_big = np.frombuffer(endian_test, dtype=">u4")[0]
        endian_little = np.frombuffer(endian_test, dtype="<u4")[0]

        if endian_big == _MAGIC:
            endian = ">"
        elif endian_little == _MAGIC:
            endian = "<"
        else:
            raise ValueError("BF byte-order marker not recognized")

        version = np.fromfile(fid, dtype=endian + "u4", count=1)
        if version.size != 1 or int(version.item()) != _FORMAT_VERSION:
            raise ValueError(f"Unsupported BF format version: {version.item() if version.size else 'missing'}")

        tmp = np.fromfile(fid, dtype=endian + "f4", count=9)
        if tmp.size != 9:
            raise ValueError("Truncated BF header: expected nine metadata values")

        header = {"fid": fid,"format_version": _FORMAT_VERSION,"bearing_frame": "relative"}

        for name, value, minimum in (("Nseq", tmp[1], 0),("Jbeams", tmp[2], 1),("nfft", tmp[3], 1),
            ("Nfreq", tmp[4], 1),("advance", tmp[5], 1),("Nchan", tmp[6], 1),("Nshades", tmp[7], 1)):
            if not np.isfinite(value) or value != np.floor(value):
                raise ValueError(
                    f"BF header {name} must be a finite integer; "
                    f"received {value!r}")

            header[name] = validate_header_count(int(value), name, minimum=minimum)

        header["sample_rate_hz"] = validate_header_positive_float32(tmp[0], "sample_rate_hz")
        header["secs_per_seq"] = validate_header_positive_float32(tmp[8], "secs_per_seq")

        # Check the stored duration against the sample rate and advance.
        expected_secs_per_seq = validate_header_positive_float32(header["advance"] / float(header["sample_rate_hz"]), "expected_secs_per_seq")
        
        if not np.isclose(header["secs_per_seq"],expected_secs_per_seq,rtol=4 * np.finfo(np.float32).eps,atol=0):
            raise ValueError("BF sequence duration does not match advance/sample_rate_hz")

        if header["Nshades"] not in (1, header["Jbeams"]):
            raise ValueError("BF Nshades must be 1 or Jbeams")

        byte_header = np.fromfile(fid, dtype=endian + "u8", count=1)
        if byte_header.size != 1:
            raise ValueError("Truncated BF header: missing header size")

        header["byte_header"] = int(byte_header.item())

        nweights = header["Nshades"] * header["Nchan"]
        ncoordinates = header["Jbeams"] * 3

        expected_byte_header = _FIXED_HEADER_BYTES + 4 * (header["nfft"] + nweights + header["Nfreq"] + ncoordinates)
        
        if header["byte_header"] != expected_byte_header:
            raise ValueError("BF header size does not match its dimensions")

        # Bound variable-length reads before allocating arrays.
        if fstat(fid.fileno()).st_size < expected_byte_header:
            raise ValueError("Truncated BF header: dimensions exceed available bytes")

        timestamp = np.fromfile(fid, dtype=endian + "u8", count=1)
        if timestamp.size != 1:
            raise ValueError("Truncated BF header: missing timestamp")

        header["start_unix_nanoseconds"] = validate_header_timestamp(timestamp.item())
        header["start_unix_seconds"] = header["start_unix_nanoseconds"] / 1e9

        dtype_code = np.fromfile(fid, dtype=np.uint8, count=1)
        if dtype_code.size != 1:
            raise ValueError("Truncated BF header: missing data type")

        try:
            header["dtype"] = code_to_dtype(int(dtype_code.item()))
        except KeyError as exc:
            raise ValueError(f"Unrecognized BF data type code: {dtype_code.item()}") from exc

        window_T = np.fromfile(fid, dtype=endian + "f4", count=header["nfft"])
        header["window_T"] = validate_header_array(window_T, "window_T", shape=(header["nfft"],))

        shade = np.fromfile(fid, dtype=endian + "f4", count=nweights)
        if shade.size != nweights:
            raise ValueError("Truncated BF header: incomplete shading data")

        header["shade"] = validate_header_shading(shade.reshape(header["Nshades"], header["Nchan"]), header["Jbeams"])

        freq = np.fromfile(fid, dtype=endian + "f4", count=header["Nfreq"])
        if freq.size != header["Nfreq"]:
            raise ValueError("Truncated BF header: incomplete frequency array")

        header["freq"] = validate_header_frequencies(freq, header["sample_rate_hz"], header["nfft"])

        steer = np.fromfile(fid, dtype=endian + "f4", count=ncoordinates)
        if steer.size != ncoordinates:
            raise ValueError("Truncated BF header: incomplete steering grid")

        header["steer_rbe_rad_J3"] = validate_header_beam_grid(steer.reshape(header["Jbeams"], 3))

        header["endian"] = endian
        
        bytes_per_record = 4 + header["Nfreq"] * header["Jbeams"] * dtype_to_bytes(header["dtype"])
        expected_file_bytes = header["byte_header"] + header["Nseq"] * bytes_per_record
        
        actual_file_bytes = fstat(fid.fileno()).st_size

        if actual_file_bytes < expected_file_bytes:
            raise ValueError(
                f"Truncated BF data: header declares {header['Nseq']} records, "
                f"requiring {expected_file_bytes} bytes; "
                f"file contains {actual_file_bytes} bytes")

        return header
    except BaseException:
        fid.close()
        raise

def bf_read_multiple_sequence(header,seqs_to_read):
    """Return ``(data_QJT, headings_T)`` for selected beam records."""

    seqs_to_read = validate_read_indices(seqs_to_read, header["Nseq"], "seqs_to_read")

    fid = header["fid"]
    Jbeams = header["Jbeams"]
    nfreq = header["Nfreq"]
    npts = len(seqs_to_read)
    dtype = header["dtype"]

    file_dtype = header["dtype"].newbyteorder(header["endian"])

    data_byte_size = dtype_to_bytes(dtype)

    data_QJT = np.empty((nfreq,Jbeams,npts),dtype=dtype)
    headings_T = np.empty(npts, dtype=np.float32)
    bytes_per_record = 4 + nfreq * Jbeams * data_byte_size

    for idx,seq in enumerate(seqs_to_read):

        # move the pointer to the right location to read
        seq = int(seq) # force it to python int which can grow without wrapping
        byte_start_seq = header["byte_header"] + seq*bytes_per_record
        fid.seek(byte_start_seq,0)

        heading = np.fromfile(fid, dtype=header["endian"] + "f4", count=1)
        if heading.size != 1:
            raise ValueError(f"Truncated BF heading at sequence {seq}")
        headings_T[idx] = validate_array_heading_deg(heading.item())
        
        temp = np.fromfile(fid,dtype=file_dtype,count=header["Nfreq"]*header["Jbeams"])
        data_QJT[:,:,idx] = temp.reshape((header["Nfreq"],header["Jbeams"]))

    # Preserve frequency, beam, and sequence axes, including singleton axes.
    return data_QJT, headings_T

def bf_read_single_sequence(header, seq_to_read):

    """Return ``(data_QJ, heading_deg)`` for one beam record."""

    data_QJT, headings_T = bf_read_multiple_sequence(header,(seq_to_read,))
    
    return data_QJT[:, :, 0], float(headings_T[0])

def bf_overwrite_Nseq(header,Nseq):
    Nseq = validate_header_count(Nseq, "Nseq", minimum=0)
    header["fid"].seek(4+4+4,0)
    np.asarray(Nseq, dtype=np.float32).tofile(header["fid"])
