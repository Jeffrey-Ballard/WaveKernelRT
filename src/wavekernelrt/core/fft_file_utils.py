"""Binary I/O for channel frequency-domain (FFT/PSD) files."""

from os import fstat

import numpy as np

from wavekernelrt.core.data_types import dtype_to_code, code_to_dtype, dtype_to_bytes

from wavekernelrt.core.validate_utils import (validate_header_count,validate_header_positive_float32,
                                              validate_header_timestamp,validate_header_array, validate_header_frequencies,
                                              validate_read_indices)

_MAGIC = 0x12345678
_FORMAT_VERSION = 1
_FIXED_HEADER_BYTES = 4 + 4 + 6 * 4 + 8 + 8 + 1

def fft_write_header(filename,Nseq,sample_rate_hz,start_unix_nanoseconds,nfft,advance,freq,channels,window_T,dtype):

    Nseq = validate_header_count(Nseq, "Nseq", minimum=0)
    nfft = validate_header_count(nfft, "nfft")
    advance = validate_header_count(advance, "advance")

    sample_rate_hz = validate_header_positive_float32(sample_rate_hz, "sample_rate_hz" )

    start_unix_nanoseconds = validate_header_timestamp(start_unix_nanoseconds)

    channels = validate_header_array(channels, "channels", shape=(None,))
    freq = validate_header_frequencies(freq, sample_rate_hz, nfft)
    window_T = validate_header_array(window_T, "window_T", shape=(nfft,))

    Nchan = validate_header_count(channels.size, "Nchan")
    Nfreq = validate_header_count(freq.size, "Nfreq")

    # Validate the data type before opening/truncating the output.
    dtype = np.dtype(dtype)
    dtype_code = dtype_to_code(dtype)

    # make sure to close it out if it fails on construction
    fid = open(filename,"wb")
    try:      
        header = dict()
        header["fid"] = fid
        header["format_version"] = _FORMAT_VERSION
        header["sample_rate_hz"] = sample_rate_hz
        header["Nseq"] = Nseq
        header["Nchan"] = Nchan
        header["nfft"] = nfft
        header["Nfreq"] = Nfreq
        header["advance"] = advance
        header["window_T"] = window_T
        header["freq"] = freq
        header["channels"] = channels
        header["dtype"] = dtype

        header["start_unix_seconds"] = start_unix_nanoseconds/1e9
        header["start_unix_nanoseconds"] = start_unix_nanoseconds # to be consistent with reading header
        
        header["secs_per_seq"] = advance/sample_rate_hz

        header["byte_header"] = _FIXED_HEADER_BYTES + 4 * (header["Nfreq"] + header["nfft"] + header["Nchan"])

        np.array([_MAGIC],dtype=np.uint32).tofile(fid)
        np.array([_FORMAT_VERSION],dtype=np.uint32).tofile(fid)
        np.array([sample_rate_hz],dtype=np.float32).tofile(fid)
        np.array([Nseq],dtype=np.float32).tofile(fid)
        np.array(header["Nchan"],dtype=np.float32).tofile(fid)
        np.array([nfft],dtype=np.float32).tofile(fid)
        np.array([header["Nfreq"]],dtype=np.float32).tofile(fid)
        np.array([advance],dtype=np.float32).tofile(fid)
        np.array([header["byte_header"]], dtype=np.uint64).tofile(fid)
        np.array([start_unix_nanoseconds], dtype=np.uint64).tofile(fid)
        np.array([dtype_code],dtype=np.uint8).tofile(fid)
        np.array([window_T],dtype=np.float32).tofile(fid)
        np.array([freq],dtype=np.float32).tofile(fid)
        np.array([channels],dtype=np.float32).tofile(fid)
    
        return header
    except BaseException:
        fid.close()
        raise

def fft_write_sequence(header,Y_QN, *, check_finite=False):

    Y_QN = np.asarray(Y_QN)

    expected_shape = (header["Nfreq"], header["Nchan"])
    if Y_QN.shape != expected_shape:
        raise ValueError(
            f"FFT record must have shape {expected_shape}; "
            f"received {Y_QN.shape}")

    output_dtype = np.dtype(header["dtype"])

    if Y_QN.dtype.kind not in "iufc":
        raise ValueError("FFT record must contain real or complex numeric values")

    if not np.can_cast(Y_QN.dtype, output_dtype, casting="same_kind"):
        raise ValueError(
            f"Cannot convert FFT record from {Y_QN.dtype} to {output_dtype} "
            "using same-kind casting")

    if check_finite and not np.all(np.isfinite(Y_QN)):
        raise ValueError("FFT record values must be finite")

    data_out = Y_QN.astype(output_dtype, copy=False)

    if check_finite and not np.all(np.isfinite(data_out)):
        raise ValueError("FFT record values must remain finite after conversion")

    data_out.tofile(header["fid"])

def fft_read_header(filename):
    
    fid = open(filename,"rb")
    
    try:

        endian_test = fid.read(4)
        if len(endian_test) != 4:
            raise ValueError("Truncated FFT header: missing byte-order marker")

        endian_big = np.frombuffer(endian_test, dtype=">u4")[0]
        endian_little = np.frombuffer(endian_test, dtype="<u4")[0]

        if endian_big == _MAGIC:
            endian = ">"
        elif endian_little == _MAGIC:
            endian = "<"
        else:
            raise ValueError("FFT byte-order marker not recognized")

        version = np.fromfile(fid, dtype=endian + "u4", count=1)
        if version.size != 1 or int(version.item()) != _FORMAT_VERSION:
            received = version.item() if version.size else "missing"
            raise ValueError(f"Unsupported FFT format version: {received}")

        tmp = np.fromfile(fid, dtype=endian + "f4", count=6)
        if tmp.size != 6:
            raise ValueError("Truncated FFT header: expected six metadata values")

        header = {"fid": fid, "format_version": _FORMAT_VERSION}

        for name, value, minimum in (("Nseq", tmp[1], 0),("Nchan", tmp[2], 1),("nfft", tmp[3], 1),("Nfreq", tmp[4], 1),
            ("advance", tmp[5], 1)):
            if not np.isfinite(value) or value != np.floor(value):
                raise ValueError(
                    f"FFT header {name} must be a finite integer; "
                    f"received {value!r}")

            header[name] = validate_header_count(int(value), name, minimum=minimum)

        header["sample_rate_hz"] = validate_header_positive_float32(tmp[0], "sample_rate_hz")

        byte_header = np.fromfile(fid, dtype=endian + "u8", count=1)
        if byte_header.size != 1:
            raise ValueError("Truncated FFT header: missing header size")

        header["byte_header"] = int(byte_header.item())

        expected_byte_header = _FIXED_HEADER_BYTES + 4 * (header["nfft"] + header["Nfreq"] + header["Nchan"])

        if header["byte_header"] != expected_byte_header:
            raise ValueError("FFT header size does not match its dimensions")

        # Bound variable-length reads before allocating arrays.
        if fstat(fid.fileno()).st_size < expected_byte_header:
            raise ValueError("Truncated FFT header: dimensions exceed available bytes")

        timestamp = np.fromfile(fid, dtype=endian + "u8", count=1)
        if timestamp.size != 1:
            raise ValueError("Truncated FFT header: missing timestamp")

        header["start_unix_nanoseconds"] = validate_header_timestamp(timestamp.item())
        header["start_unix_seconds"] = header["start_unix_nanoseconds"] / 1e9

        dtype_code = np.fromfile(fid, dtype=np.uint8, count=1)
        if dtype_code.size != 1:
            raise ValueError("Truncated FFT header: missing data type")

        try:
            header["dtype"] = code_to_dtype(int(dtype_code.item()))
        except KeyError as exc:
            raise ValueError(f"Unrecognized FFT data type code: {dtype_code.item()}") from exc

        win = np.fromfile(fid, dtype=endian + "f4", count=header["nfft"])
        header["window_T"] = validate_header_array(win, "window_T", shape=(header["nfft"],))

        freq = np.fromfile(fid, dtype=endian + "f4", count=header["Nfreq"])
        if freq.size != header["Nfreq"]:
            raise ValueError("Truncated FFT header: incomplete frequency array")

        header["freq"] = validate_header_frequencies(freq, header["sample_rate_hz"], header["nfft"])

        channels = np.fromfile(fid, dtype=endian + "f4", count=header["Nchan"])
        header["channels"] = validate_header_array(channels, "channels", shape=(header["Nchan"],))

        header["endian"] = endian

        header["secs_per_seq"] = header["advance"]/header["sample_rate_hz"]

        bytes_per_record = header["Nfreq"] * header["Nchan"] * dtype_to_bytes(header["dtype"])
        expected_file_bytes = header["byte_header"] + header["Nseq"] * bytes_per_record
        
        actual_file_bytes = fstat(fid.fileno()).st_size

        if actual_file_bytes < expected_file_bytes:
            raise ValueError(
                f"Truncated FFT data: header declares {header['Nseq']} records, "
                f"requiring {expected_file_bytes} bytes; "
                f"file contains {actual_file_bytes} bytes")

        return header
    except BaseException:
        fid.close()
        raise

def fft_read_multiple_sequence(header,seqs_to_read):

    seqs_to_read = validate_read_indices(seqs_to_read, header["Nseq"], "seqs_to_read")

    fid = header["fid"]
    nchan = header["Nchan"]
    nfreq = header["Nfreq"]
    npts = len(seqs_to_read)

    # Grab the dtype info to be able to read things properly
    file_dtype = header["dtype"].newbyteorder(header["endian"])
    dtype = header["dtype"]
    data_byte_size = dtype_to_bytes(dtype)

    # we want QN to be continguous in memory for each seq and each N continguous
    # for every Q to send to the BF downstream
    data_TQN = np.empty((npts,nfreq,nchan),dtype=dtype)
    
    bytes_per_seq = nfreq*nchan*data_byte_size
    for idx_t,seq in enumerate(seqs_to_read):

        # move the pointer to the right location to read
        seq = int(seq) # force it to python int which can grow without wrapping
        byte_start_seq = header["byte_header"] + seq*bytes_per_seq
        fid.seek(byte_start_seq,0)
        
        temp = np.fromfile(fid,dtype=file_dtype,count=nfreq*nchan)
        data_TQN[idx_t,:,:] = temp.reshape((nfreq,nchan))

    return data_TQN

def fft_read_single_sequence(header,seq_to_read):

    """Read exactly one record as (frequency, channel)."""
    data_TQN = fft_read_multiple_sequence(header,(seq_to_read,))
    return data_TQN[0]

def fft_overwrite_Nseq(header,Nseq):
    Nseq = validate_header_count(Nseq, "Nseq", minimum=0)
    header["fid"].seek(4+4+4,0)
    np.asarray(Nseq, dtype=np.float32).tofile(header["fid"])
