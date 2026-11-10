"""Read and write white-noise-gain (WNG) sidecar files."""

from os import fstat

import numpy as np

from wavekernelrt.core.bf_file_utils import bf_read_multiple_sequence,bf_read_single_sequence,bf_write_sequence
from wavekernelrt.core.data_types import code_to_dtype, dtype_to_code
from wavekernelrt.core.validate_utils import (validate_header_array,validate_header_beam_grid,validate_header_count,
    validate_header_float32,validate_header_frequencies,validate_header_positive_float32,validate_header_shading,
    validate_header_timestamp)

_MAGIC = 0x12345678
_FORMAT_VERSION = 1
_FIXED_HEADER_BYTES = 4 + 4 + 11 * 4 + 8 + 8 + 1
_DTYPE = np.dtype(np.float32)


def wng_write_header(filename,Nseq,sample_rate_hz,start_unix_nanoseconds,nfft,advance,freq,steer_rbe_rad_J3,window_T,
    shade,wngc_val_dB,wngc_tol_dB):
    """Create a WNG file and return its open header dictionary."""

    Nseq = validate_header_count(Nseq, "Nseq", minimum=0)
    nfft = validate_header_count(nfft, "nfft")
    advance = validate_header_count(advance, "advance")
    sample_rate_hz = validate_header_positive_float32(sample_rate_hz,"sample_rate_hz")
    start_unix_nanoseconds = validate_header_timestamp(start_unix_nanoseconds)

    freq = validate_header_frequencies(freq, sample_rate_hz, nfft)
    window_T = validate_header_array(window_T, "window_T", shape=(nfft,))
    steer_rbe_rad_J3 = validate_header_beam_grid(steer_rbe_rad_J3)
    Jbeams = steer_rbe_rad_J3.shape[0]
    shade = validate_header_shading(shade, Jbeams)

    Nfreq = validate_header_count(freq.size, "Nfreq")
    Nshades, Nchan = shade.shape
    secs_per_seq = validate_header_positive_float32(advance / float(sample_rate_hz),"secs_per_seq")
    wngc_val_dB = validate_header_float32(wngc_val_dB,"wngc_val_dB")
    wngc_tol_dB = validate_header_float32(wngc_tol_dB,"wngc_tol_dB",nonnegative=True)

    byte_header = _FIXED_HEADER_BYTES + _DTYPE.itemsize * (nfft + Nshades * Nchan + Nfreq + Jbeams * 3)

    fid = open(filename, "wb")
    try:
        header = {"fid": fid,"format_version": _FORMAT_VERSION,"bearing_frame": "relative",
            "sample_rate_hz": sample_rate_hz,"Nseq": int(Nseq),"Jbeams": int(Jbeams),"nfft": int(nfft),
            "window_T": np.asarray(window_T, dtype=np.float32),"Nfreq": int(Nfreq),"advance": int(advance),
            "Nshades": int(Nshades),"Nchan": int(Nchan),"shade": shade,"freq": np.asarray(freq, dtype=np.float32),
            "steer_rbe_rad_J3": np.asarray(steer_rbe_rad_J3,dtype=np.float32),"dtype": _DTYPE,
            "start_unix_seconds": start_unix_nanoseconds / 1e9,"start_unix_nanoseconds": int(start_unix_nanoseconds),
            "secs_per_seq": secs_per_seq,"wngc_val_dB": wngc_val_dB,"wngc_tol_dB": wngc_tol_dB,
            "byte_header": byte_header}

        np.asarray(_MAGIC, dtype=np.uint32).tofile(fid)
        np.asarray(_FORMAT_VERSION, dtype=np.uint32).tofile(fid)
        np.asarray([sample_rate_hz,Nseq,Jbeams,nfft,Nfreq,advance,Nchan,Nshades,secs_per_seq,wngc_val_dB,wngc_tol_dB],
            dtype=np.float32).tofile(fid)
        np.asarray(byte_header, dtype=np.uint64).tofile(fid)
        np.asarray(start_unix_nanoseconds, dtype=np.uint64).tofile(fid)
        np.asarray(dtype_to_code(_DTYPE), dtype=np.uint8).tofile(fid)

        header["window_T"].tofile(fid)
        header["shade"].tofile(fid)
        header["freq"].tofile(fid)
        header["steer_rbe_rad_J3"].tofile(fid)

        return header
    except BaseException:
        fid.close()
        raise


def wng_read_header(filename):
    """Open a WNG file and return its validated header dictionary."""

    fid = open(filename, "rb")
    try:
        endian_test = fid.read(4)
        if len(endian_test) != 4:
            raise ValueError("Truncated WNG header: missing byte-order marker")

        endian_big = np.frombuffer(endian_test, dtype=">u4")[0]
        endian_little = np.frombuffer(endian_test, dtype="<u4")[0]
        if endian_big == _MAGIC:
            endian = ">"
        elif endian_little == _MAGIC:
            endian = "<"
        else:
            raise ValueError("WNG byte-order marker not recognized")

        version = np.fromfile(fid, dtype=endian + "u4", count=1)
        if version.size != 1 or int(version.item()) != _FORMAT_VERSION:
            received = version.item() if version.size else "missing"
            raise ValueError(f"Unsupported WNG format version: {received}")

        metadata = np.fromfile(fid, dtype=endian + "f4", count=11)
        if metadata.size != 11:
            raise ValueError("Truncated WNG header: expected eleven metadata values")

        header = {"fid": fid,"format_version": _FORMAT_VERSION,"bearing_frame": "relative"}
        for name, value, minimum in (("Nseq", metadata[1], 0),("Jbeams", metadata[2], 1),("nfft", metadata[3], 1),
            ("Nfreq", metadata[4], 1),("advance", metadata[5], 1),("Nchan", metadata[6], 1),
            ("Nshades", metadata[7], 1)):
            if not np.isfinite(value) or value != np.floor(value):
                raise ValueError(f"WNG header {name} must be a finite integer; received {value!r}")
            header[name] = validate_header_count(int(value),name,minimum=minimum)

        header["sample_rate_hz"] = validate_header_positive_float32(metadata[0],"sample_rate_hz")
        header["secs_per_seq"] = validate_header_positive_float32(metadata[8],"secs_per_seq")
        header["wngc_val_dB"] = validate_header_float32(metadata[9],"wngc_val_dB")
        header["wngc_tol_dB"] = validate_header_float32(metadata[10],"wngc_tol_dB",nonnegative=True)

        expected_secs_per_seq = validate_header_positive_float32(header["advance"] / float(header["sample_rate_hz"]),
            "expected_secs_per_seq")

        if not np.isclose(header["secs_per_seq"],expected_secs_per_seq,rtol=4 * np.finfo(np.float32).eps,atol=0):
            raise ValueError("WNG sequence duration does not match advance/sample_rate_hz")
        if header["Nshades"] not in (1, header["Jbeams"]):
            raise ValueError("WNG Nshades must be 1 or Jbeams")

        byte_header = np.fromfile(fid, dtype=endian + "u8", count=1)
        if byte_header.size != 1:
            raise ValueError("Truncated WNG header: missing header size")
        header["byte_header"] = int(byte_header.item())

        nweights = header["Nshades"] * header["Nchan"]
        ncoordinates = header["Jbeams"] * 3
        expected_byte_header = _FIXED_HEADER_BYTES + _DTYPE.itemsize * (header["nfft"] + nweights + header["Nfreq"] + ncoordinates)
        if header["byte_header"] != expected_byte_header:
            raise ValueError("WNG header size does not match its dimensions")
        if fstat(fid.fileno()).st_size < expected_byte_header:
            raise ValueError("Truncated WNG header: dimensions exceed available bytes")

        timestamp = np.fromfile(fid, dtype=endian + "u8", count=1)
        if timestamp.size != 1:
            raise ValueError("Truncated WNG header: missing timestamp")
        header["start_unix_nanoseconds"] = validate_header_timestamp(timestamp.item())
        header["start_unix_seconds"] = header["start_unix_nanoseconds"] / 1e9

        dtype_code = np.fromfile(fid, dtype=np.uint8, count=1)
        if dtype_code.size != 1:
            raise ValueError("Truncated WNG header: missing data type")
        try:
            header["dtype"] = code_to_dtype(int(dtype_code.item()))
        except KeyError as exc:
            raise ValueError(f"Unrecognized WNG data type code: {dtype_code.item()}") from exc
        if header["dtype"] != _DTYPE:
            raise ValueError("WNG record dtype must be float32")

        window_T = np.fromfile(fid, dtype=endian + "f4", count=header["nfft"])
        header["window_T"] = validate_header_array(window_T,"window_T",shape=(header["nfft"],))

        shade = np.fromfile(fid, dtype=endian + "f4", count=nweights)
        if shade.size != nweights:
            raise ValueError("Truncated WNG header: incomplete shading data")
        header["shade"] = validate_header_shading(shade.reshape(header["Nshades"], header["Nchan"]),header["Jbeams"])

        freq = np.fromfile(fid, dtype=endian + "f4", count=header["Nfreq"])
        if freq.size != header["Nfreq"]:
            raise ValueError("Truncated WNG header: incomplete frequency array")
        header["freq"] = validate_header_frequencies(freq,header["sample_rate_hz"],header["nfft"])

        steer = np.fromfile(fid, dtype=endian + "f4", count=ncoordinates)
        if steer.size != ncoordinates:
            raise ValueError("Truncated WNG header: incomplete steering grid")
        header["steer_rbe_rad_J3"] = validate_header_beam_grid(steer.reshape(header["Jbeams"], 3))
        header["endian"] = endian

        bytes_per_record = _DTYPE.itemsize * (1 + header["Nfreq"] * header["Jbeams"])
        expected_file_bytes = (header["byte_header"] + header["Nseq"] * bytes_per_record)
        if fstat(fid.fileno()).st_size < expected_file_bytes:
            raise ValueError(
                f"Truncated WNG data: header declares {header['Nseq']} records, "
                f"requiring {expected_file_bytes} bytes")

        return header
    except BaseException:
        fid.close()
        raise


def wng_write_sequence(header, wng_QJ, array_heading_deg, *, check_finite=False):
    """Write one ``(Nfreq, Jbeams)`` linear-WNG record."""

    return bf_write_sequence(header,wng_QJ,array_heading_deg,check_finite=check_finite)


def wng_read_multiple_sequence(header, seqs_to_read):
    """Return ``(wng_QJT, headings_T)`` for selected WNG records."""

    return bf_read_multiple_sequence(header, seqs_to_read)


def wng_read_single_sequence(header, seq_to_read):
    """Return ``(wng_QJ, heading_deg)`` for one WNG record."""

    return bf_read_single_sequence(header, seq_to_read)


def wng_constraint_linear_J(header):
    """Return the effective linear WNG constraint for every stored beam.

    DMR constrains WNG relative to the unshaded conventional-beamformer WNG.
    Current steering replicas have unit magnitude, so that reference equals
    the number of active (nonzero-shaded) elements in each beam aperture.
    ``wngc_tol_dB`` relaxes the requested relative value when determining
    feasibility.
    """
    shade = np.asarray(header["shade"])
    if shade.ndim != 2 or shade.shape[0] not in (1, header["Jbeams"]):
        raise ValueError("WNG header shade must have one row or one row per beam")

    active_count = np.count_nonzero(shade > 0, axis=1)
    if active_count.size == 1:
        active_count = np.full(header["Jbeams"], active_count.item())

    relative_constraint = 10.0 ** ((float(header["wngc_val_dB"]) - float(header["wngc_tol_dB"])) / 10.0)
    return active_count.astype(np.float64) * relative_constraint


def wng_overwrite_Nseq(header, Nseq):
    """Overwrite the record count in an open WNG header."""

    Nseq = validate_header_count(Nseq, "Nseq", minimum=0)
    header["fid"].seek(4 + 4 + 4, 0)
    np.asarray(Nseq, dtype=np.float32).tofile(header["fid"])
