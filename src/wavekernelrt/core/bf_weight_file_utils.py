"""Binary I/O for time-varying beamformer multiplication weights.

Each record is stored in ``(frequency, beam, element)`` (QJN) order.  The
values are the already-conjugated multiplication coefficients applied to an
FFT input, so a stored record satisfies::

    beamformed_QJ = np.einsum("qn,qjn->qj", data_QN, weights_QJN)

In particular, if an algorithm derives a mathematical weight vector ``w``
and applies it as ``x @ conj(w)``, this format stores ``conj(w)``.
"""

from os import fstat

import numpy as np

from wavekernelrt.core.data_types import code_to_dtype,dtype_to_bytes,dtype_to_code
from wavekernelrt.core.validate_utils import (validate_header_array,validate_header_beam_grid,validate_header_count,
    validate_header_frequencies,validate_header_positive_float32,validate_header_timestamp,validate_read_indices,
    validate_array_heading_deg)


_MAGIC = 0x12345678
_FORMAT_VERSION = 1
_FIXED_HEADER_BYTES = 4 + 4 + 8 * 4 + 8 + 8 + 1


def bf_weight_write_header(filename,Nseq,sample_rate_hz,start_unix_nanoseconds,nfft,advance,freq,channels,
    steer_rbe_rad_J3,window_T,dtype=np.complex64):
    """Create a beamformer-weight file and return its open header dictionary.

    Version 1 records contain one array heading followed by an
    ``(Nfreq, Jbeams, Nchan)`` array of already-conjugated coefficients used
    to multiply ``data_QN``.
    """

    Nseq = validate_header_count(Nseq, "Nseq", minimum=0)
    nfft = validate_header_count(nfft, "nfft")
    advance = validate_header_count(advance, "advance")
    sample_rate_hz = validate_header_positive_float32(sample_rate_hz,"sample_rate_hz")
    start_unix_nanoseconds = validate_header_timestamp(start_unix_nanoseconds)

    freq = validate_header_frequencies(freq, sample_rate_hz, nfft)
    channels = validate_header_array(channels, "channels", shape=(None,))
    steer_rbe_rad_J3 = validate_header_beam_grid(steer_rbe_rad_J3)
    window_T = validate_header_array(window_T, "window_T", shape=(nfft,))

    Nfreq = validate_header_count(freq.size, "Nfreq")
    Nchan = validate_header_count(channels.size, "Nchan")
    Jbeams = validate_header_count(steer_rbe_rad_J3.shape[0], "Jbeams")
    secs_per_seq = validate_header_positive_float32(advance / float(sample_rate_hz),"secs_per_seq")

    dtype = np.dtype(dtype)
    if dtype.kind != "c":
        raise ValueError("BF weight dtype must be a complex dtype")
    dtype_code = dtype_to_code(dtype)

    byte_header = _FIXED_HEADER_BYTES + 4 * (nfft + Nchan + Nfreq + 3 * Jbeams)

    fid = open(filename, "wb")
    try:
        header = {"fid": fid,"format_version": _FORMAT_VERSION,"sample_rate_hz": sample_rate_hz,"Nseq": Nseq,
            "Jbeams": Jbeams,"Nchan": Nchan,"nfft": nfft,"Nfreq": Nfreq,"advance": advance,
            "secs_per_seq": secs_per_seq,"byte_header": byte_header,"start_unix_nanoseconds": start_unix_nanoseconds,
            "start_unix_seconds": start_unix_nanoseconds / 1e9,"dtype": dtype,"window_T": window_T,
            "channels": channels,"freq": freq,"steer_rbe_rad_J3": steer_rbe_rad_J3,"axis_order": "QJN",
            "weight_convention": "conjugated_multiplication_coefficients","bearing_frame": "relative"}

        np.asarray([_MAGIC], dtype=np.uint32).tofile(fid)
        np.asarray([_FORMAT_VERSION], dtype=np.uint32).tofile(fid)
        np.asarray([sample_rate_hz,Nseq,Jbeams,Nchan,nfft,Nfreq,advance,secs_per_seq],dtype=np.float32).tofile(fid)
        np.asarray([byte_header], dtype=np.uint64).tofile(fid)
        np.asarray([start_unix_nanoseconds], dtype=np.uint64).tofile(fid)
        np.asarray([dtype_code], dtype=np.uint8).tofile(fid)
        window_T.tofile(fid)
        channels.tofile(fid)
        freq.tofile(fid)
        steer_rbe_rad_J3.tofile(fid)

        return header
    except BaseException:
        fid.close()
        raise


def bf_weight_write_sequence(header, weights_QJN, array_heading_deg, *, check_finite=False):
    """Append one ``(frequency, beam, element)`` weight record."""

    weights_QJN = np.asarray(weights_QJN)
    expected_shape = (header["Nfreq"],header["Jbeams"],header["Nchan"])
   
    if weights_QJN.shape != expected_shape:
        raise ValueError(
            f"BF weight record must have shape {expected_shape}; "
            f"received {weights_QJN.shape}")

    if weights_QJN.dtype.kind not in "iufc":
        raise ValueError("BF weight record must contain numeric values")

    if check_finite and not np.all(np.isfinite(weights_QJN)):
        raise ValueError("BF weight record values must be finite")

    data_out = weights_QJN.astype(np.dtype(header["dtype"]), copy=False)
    if check_finite and not np.all(np.isfinite(data_out)):
        raise ValueError("BF weight record values must remain finite after conversion")

    heading = validate_array_heading_deg(array_heading_deg)
    np.asarray(heading, dtype=np.float32).tofile(header["fid"])
    data_out.tofile(header["fid"])


def bf_weight_read_header(filename):
    """Read and validate a beamformer-weight header, leaving the file open."""

    fid = open(filename, "rb")
    try:
        endian_test = fid.read(4)
        if len(endian_test) != 4:
            raise ValueError("Truncated BF weight header: missing byte-order marker")

        endian_big = np.frombuffer(endian_test, dtype=">u4")[0]
        endian_little = np.frombuffer(endian_test, dtype="<u4")[0]
        if endian_big == _MAGIC:
            endian = ">"
        elif endian_little == _MAGIC:
            endian = "<"
        else:
            raise ValueError("BF weight byte-order marker not recognized")

        version = np.fromfile(fid, dtype=endian + "u4", count=1)
        if version.size != 1:
            raise ValueError("Truncated BF weight header: missing version")
        if int(version.item()) != _FORMAT_VERSION:
            raise ValueError(
                "Unsupported BF weight format version: "
                f"{int(version.item())}")

        metadata = np.fromfile(fid, dtype=endian + "f4", count=8)
        if metadata.size != 8:
            raise ValueError("Truncated BF weight header: expected eight metadata values")

        header = {"fid": fid, "format_version": _FORMAT_VERSION}
        for name, value, minimum in (("Nseq", metadata[1], 0),("Jbeams", metadata[2], 1),("Nchan", metadata[3], 1),
            ("nfft", metadata[4], 1),("Nfreq", metadata[5], 1),("advance", metadata[6], 1)):
            if not np.isfinite(value) or value != np.floor(value):
                raise ValueError(
                    f"BF weight header {name} must be a finite integer; "
                    f"received {value!r}")
            header[name] = validate_header_count(int(value),name,minimum=minimum)

        header["sample_rate_hz"] = validate_header_positive_float32(metadata[0],"sample_rate_hz")
        header["secs_per_seq"] = validate_header_positive_float32(metadata[7],"secs_per_seq")
        expected_secs_per_seq = validate_header_positive_float32(header["advance"] / float(header["sample_rate_hz"]),
            "expected_secs_per_seq")
        if not np.isclose(header["secs_per_seq"],expected_secs_per_seq,rtol=4 * np.finfo(np.float32).eps,atol=0):
            raise ValueError("BF weight sequence duration does not match advance/sample_rate_hz")

        byte_header = np.fromfile(fid, dtype=endian + "u8", count=1)
        if byte_header.size != 1:
            raise ValueError("Truncated BF weight header: missing header size")
        header["byte_header"] = int(byte_header.item())

        expected_byte_header = _FIXED_HEADER_BYTES + 4 * (header["nfft"] + header["Nchan"] + header["Nfreq"] + 3 * header["Jbeams"])
        
        if header["byte_header"] != expected_byte_header:
            raise ValueError("BF weight header size does not match its dimensions")
        
        if fstat(fid.fileno()).st_size < expected_byte_header:
            raise ValueError("Truncated BF weight header: dimensions exceed available bytes")

        timestamp = np.fromfile(fid, dtype=endian + "u8", count=1)
        if timestamp.size != 1:
            raise ValueError("Truncated BF weight header: missing timestamp")
        header["start_unix_nanoseconds"] = validate_header_timestamp(timestamp.item())
        header["start_unix_seconds"] = (header["start_unix_nanoseconds"] / 1e9)

        dtype_code = np.fromfile(fid, dtype=np.uint8, count=1)
        if dtype_code.size != 1:
            raise ValueError("Truncated BF weight header: missing data type")
        try:
            header["dtype"] = code_to_dtype(int(dtype_code.item()))
        except KeyError as exc:
            raise ValueError(
                "Unrecognized BF weight data type code: "
                f"{dtype_code.item()}") from exc
        
        if header["dtype"].kind != "c":
            raise ValueError("BF weight header dtype must be complex")

        window_T = np.fromfile(fid,dtype=endian + "f4",count=header["nfft"])
        header["window_T"] = validate_header_array(window_T,"window_T",shape=(header["nfft"],))

        channels = np.fromfile(fid,dtype=endian + "f4",count=header["Nchan"])
        header["channels"] = validate_header_array(channels,"channels",shape=(header["Nchan"],))

        freq = np.fromfile(fid,dtype=endian + "f4",count=header["Nfreq"])
        if freq.size != header["Nfreq"]:
            raise ValueError("Truncated BF weight header: incomplete frequency array")
        
        header["freq"] = validate_header_frequencies(freq,header["sample_rate_hz"],header["nfft"])

        coordinate_count = 3 * header["Jbeams"]
        steer = np.fromfile(fid,dtype=endian + "f4",count=coordinate_count)
        if steer.size != coordinate_count:
            raise ValueError("Truncated BF weight header: incomplete steering grid")
        
        header["steer_rbe_rad_J3"] = validate_header_beam_grid(steer.reshape(header["Jbeams"], 3))

        header["endian"] = endian
        header["axis_order"] = "QJN"
        header["weight_convention"] = ("conjugated_multiplication_coefficients")
        header["bearing_frame"] = "relative"

        bytes_per_record = 4 + (header["Nfreq"] * header["Jbeams"] * header["Nchan"] * dtype_to_bytes(header["dtype"]))
        expected_file_bytes = (header["byte_header"] + header["Nseq"] * bytes_per_record)
        actual_file_bytes = fstat(fid.fileno()).st_size
        if actual_file_bytes < expected_file_bytes:
            raise ValueError(
                "Truncated BF weight data: header declares "
                f"{header['Nseq']} records, requiring "
                f"{expected_file_bytes} bytes; file contains "
                f"{actual_file_bytes} bytes")

        return header
    except BaseException:
        fid.close()
        raise


def bf_weight_read_multiple_sequence(header, seqs_to_read):
    """Return ``(weights_QJNT, headings_T)`` for selected records."""

    seqs_to_read = validate_read_indices(seqs_to_read,header["Nseq"],"seqs_to_read")

    nfreq = header["Nfreq"]
    nbeams = header["Jbeams"]
    nchan = header["Nchan"]
    nseq = len(seqs_to_read)
    dtype = header["dtype"]
    file_dtype = dtype.newbyteorder(header["endian"])
    values_per_record = nfreq * nbeams * nchan
    bytes_per_record = 4 + values_per_record * dtype_to_bytes(dtype)

    weights_QJNT = np.empty((nfreq, nbeams, nchan, nseq),dtype=dtype)
    headings_T = np.empty(nseq, dtype=np.float32)
    for idx_t, seq in enumerate(seqs_to_read):
        seq = int(seq)
        header["fid"].seek(header["byte_header"] + seq * bytes_per_record,0)
        heading = np.fromfile(header["fid"],dtype=header["endian"] + "f4",count=1)
        
        if heading.size != 1:
            raise ValueError(f"Truncated BF weight heading at sequence {seq}")
        
        headings_T[idx_t] = validate_array_heading_deg(heading.item())
        record = np.fromfile(header["fid"],dtype=file_dtype,count=values_per_record)

        if record.size != values_per_record:
            raise ValueError(f"Truncated BF weight record at sequence {seq}")
        weights_QJNT[:, :, :, idx_t] = record.reshape((nfreq, nbeams, nchan))

    return weights_QJNT, headings_T


def bf_weight_read_single_sequence(header, seq_to_read):
    """Return ``(weights_QJN, heading_deg)`` for exactly one record."""

    weights_QJNT, headings_T = bf_weight_read_multiple_sequence(header,(seq_to_read,))
    return weights_QJNT[:, :, :, 0], float(headings_T[0])


def bf_weight_overwrite_Nseq(header, Nseq):
    """Overwrite the declared record count in an open weight file."""

    Nseq = validate_header_count(Nseq, "Nseq", minimum=0)
    # magic u4, version u4, sample_rate_hz f4, then Nseq f4
    header["fid"].seek(12, 0)
    np.asarray(Nseq, dtype=np.float32).tofile(header["fid"])
    header["Nseq"] = Nseq
