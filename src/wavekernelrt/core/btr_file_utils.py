"""Binary I/O for band-energy bearing-time-record (BTR) files."""

from os import fstat

import numpy as np

from wavekernelrt.core.data_types import dtype_to_code, code_to_dtype, dtype_to_bytes
from wavekernelrt.core.validate_utils import validate_header_count, validate_header_positive_float32, validate_header_timestamp
from wavekernelrt.core.validate_utils import validate_header_bands, validate_header_axis, validate_read_indices, validate_array_heading_deg

_MAGIC = 0x12345678
_FORMAT_VERSION = 1
_FIXED_HEADER_BYTES = 4 + 4 + 8 * 4 + 8 + 8 + 1

def btr_write_header(filename,Nseq,sample_rate_hz,start_unix_nanoseconds,nfft,advance,bands_B2,azim_deg_A,elev_deg_E,dtype):

    Nseq = validate_header_count(Nseq, "Nseq", minimum=0)
    nfft = validate_header_count(nfft, "nfft")
    advance = validate_header_count(advance, "advance")

    sample_rate_hz = validate_header_positive_float32(sample_rate_hz, "sample_rate_hz")
    start_unix_nanoseconds = validate_header_timestamp(start_unix_nanoseconds)

    bands_array = validate_header_bands(bands_B2)
    azim_deg_A = validate_header_axis(azim_deg_A, "azim_deg_A")
    elev_deg_E = validate_header_axis(elev_deg_E, "elev_deg_E")

    Nbands = validate_header_count(bands_array.shape[0], "Nbands")
    Nazim = validate_header_count(azim_deg_A.size, "Nazim")
    Nelev = validate_header_count(elev_deg_E.size, "Nelev")

    secs_per_seq = validate_header_positive_float32(advance / float(sample_rate_hz), "secs_per_seq")

    byte_header = _FIXED_HEADER_BYTES + 4 * (Nbands * 2 + Nazim + Nelev)

    dtype = np.dtype(dtype)
    dtype_code = dtype_to_code(dtype)

    # make sure to close it out if it fails on construction
    fid = open(filename,"wb")
    try:

        header = {"fid": fid,"format_version": _FORMAT_VERSION,"bearing_frame": "relative",
            "sample_rate_hz": sample_rate_hz,"Nseq": Nseq,"nfft": nfft,"Nazim": Nazim,"Nelev": Nelev,"Nbands": Nbands,
            "advance": advance,"dtype": dtype,"bands_B2": [tuple(row) for row in bands_array],"azim_deg_A": azim_deg_A,
            "elev_deg_E": elev_deg_E,"start_unix_seconds": start_unix_nanoseconds / 1e9,
            "start_unix_nanoseconds": start_unix_nanoseconds,"secs_per_seq": secs_per_seq,"byte_header": byte_header}

        np.array([_MAGIC],dtype=np.uint32).tofile(fid)
        np.array([_FORMAT_VERSION],dtype=np.uint32).tofile(fid)
        np.array([sample_rate_hz],dtype=np.float32).tofile(fid)
        np.array([Nseq],dtype=np.float32).tofile(fid)
        np.array([nfft],dtype=np.float32).tofile(fid)
        np.array(header["Nbands"],dtype=np.float32).tofile(fid)
        np.array(header["Nazim"],dtype=np.float32).tofile(fid)
        np.array(header["Nelev"],dtype=np.float32).tofile(fid)
        np.array([advance],dtype=np.float32).tofile(fid)
        np.array(header["secs_per_seq"],dtype=np.float32).tofile(fid)
        np.array([header["byte_header"]], dtype=np.uint64).tofile(fid)
        np.array([start_unix_nanoseconds], dtype=np.uint64).tofile(fid)
        np.array([dtype_code],dtype=np.uint8).tofile(fid)
        bands_array.tofile(fid)
        np.array(header["azim_deg_A"],dtype=np.float32).tofile(fid)
        np.array(header["elev_deg_E"],dtype=np.float32).tofile(fid)
    
        return header
    except BaseException:
        fid.close()
        raise

def btr_read_header(filename):
    
    fid = open(filename,"rb")

    try:
        endian_test = fid.read(4)

        if len(endian_test) != 4:
            raise ValueError("Truncated BTR header: missing byte-order marker")

        endian_big = np.frombuffer(endian_test, dtype=">u4")[0]
        endian_little = np.frombuffer(endian_test, dtype="<u4")[0]

        if endian_big == _MAGIC:
            endian = ">"
        elif endian_little == _MAGIC:
            endian = "<"
        else:
            raise ValueError("BTR byte-order marker not recognized")

        version = np.fromfile(fid, dtype=endian + "u4", count=1)
        if version.size != 1 or int(version.item()) != _FORMAT_VERSION:
            raise ValueError(f"Unsupported BTR format version: {version.item() if version.size else 'missing'}")

        tmp = np.fromfile(fid, dtype=endian + "f4", count=8)
        if tmp.size != 8:
            raise ValueError("Truncated BTR header: expected eight metadata values")

        header = {"fid": fid,"format_version": _FORMAT_VERSION,"bearing_frame": "relative"}

        for name, value, minimum in (("Nseq", tmp[1], 0),("nfft", tmp[2], 1),("Nbands", tmp[3], 1),
            ("Nazim", tmp[4], 1),("Nelev", tmp[5], 1),("advance", tmp[6], 1)):
            if not np.isfinite(value) or value != np.floor(value):
                raise ValueError(
                    f"BTR header {name} must be a finite integer; "
                    f"received {value!r}")

            header[name] = validate_header_count(int(value), name, minimum=minimum)

        header["sample_rate_hz"] = validate_header_positive_float32(tmp[0], "sample_rate_hz")
        header["secs_per_seq"] = validate_header_positive_float32(tmp[7], "secs_per_seq")

        expected_secs_per_seq = validate_header_positive_float32(header["advance"] / float(header["sample_rate_hz"]),
            "expected_secs_per_seq")
        if not np.isclose(header["secs_per_seq"],expected_secs_per_seq,rtol=4 * np.finfo(np.float32).eps,atol=0):
            raise ValueError("BTR sequence duration does not match advance/sample_rate_hz")

        byte_header = np.fromfile(fid, dtype=endian + "u8", count=1)
        if byte_header.size != 1:
            raise ValueError("Truncated BTR header: missing header size")

        header["byte_header"] = int(byte_header.item())

        expected_byte_header = _FIXED_HEADER_BYTES + 4 * (header["Nbands"] * 2 + header["Nazim"] + header["Nelev"])
        
        if header["byte_header"] != expected_byte_header:
            raise ValueError("BTR header size does not match its dimensions")

        if fstat(fid.fileno()).st_size < expected_byte_header:
            raise ValueError("Truncated BTR header: dimensions exceed available bytes")

        timestamp = np.fromfile(fid, dtype=endian + "u8", count=1)
        if timestamp.size != 1:
            raise ValueError("Truncated BTR header: missing timestamp")

        header["start_unix_nanoseconds"] = validate_header_timestamp(timestamp.item())
        header["start_unix_seconds"] = header["start_unix_nanoseconds"] / 1e9

        dtype_code = np.fromfile(fid, dtype=np.uint8, count=1)
        if dtype_code.size != 1:
            raise ValueError("Truncated BTR header: missing data type")

        try:
            header["dtype"] = code_to_dtype(int(dtype_code.item()))
        except KeyError as exc:
            raise ValueError(f"Unrecognized BTR data type code: {dtype_code.item()}") from exc

        bands_array = np.fromfile(fid, dtype=endian + "f4", count=header["Nbands"] * 2)
        if bands_array.size != header["Nbands"] * 2:
            raise ValueError("Truncated BTR header: incomplete band array")

        bands_array = validate_header_bands(bands_array.reshape(header["Nbands"], 2))
        header["bands_B2"] = [tuple(row) for row in bands_array]

        azim = np.fromfile(fid, dtype=endian + "f4", count=header["Nazim"])
        if azim.size != header["Nazim"]:
            raise ValueError("Truncated BTR header: incomplete azimuth array")

        header["azim_deg_A"] = validate_header_axis(azim, "azim_deg_A")

        elev = np.fromfile(fid, dtype=endian + "f4", count=header["Nelev"])
        if elev.size != header["Nelev"]:
            raise ValueError("Truncated BTR header: incomplete elevation array")

        header["elev_deg_E"] = validate_header_axis(elev, "elev_deg_E")
                
        header["endian"] = endian

        bytes_per_record = 4 + header["Nbands"] * header["Nazim"] * header["Nelev"] * dtype_to_bytes(header["dtype"])
        expected_file_bytes = header["byte_header"] + header["Nseq"] * bytes_per_record

        actual_file_bytes = fstat(fid.fileno()).st_size

        if actual_file_bytes < expected_file_bytes:
            raise ValueError(
                f"Truncated BTR data: header declares {header['Nseq']} records, "
                f"requiring {expected_file_bytes} bytes; "
                f"file contains {actual_file_bytes} bytes")

        return header
    except BaseException:
        fid.close()
        raise

def btr_write_sequence(header,data_in,array_heading_deg, *, check_finite=False):
    data_in = np.asarray(data_in)

    if data_in.ndim == 2:
        data_BAE = data_in[:, :, np.newaxis]
    else:
        data_BAE = data_in

    expected_shape = (header["Nbands"], header["Nazim"], header["Nelev"])
    if data_BAE.shape != expected_shape:
        raise ValueError(
            f"BTR record must have shape {expected_shape}; "
            f"received {data_BAE.shape}")

    output_dtype = np.dtype(header["dtype"])

    if data_BAE.dtype.kind not in "iuf":
        raise ValueError("BTR records must contain real numeric values")

    # Integer output must not silently truncate or wrap values.
    casting = "safe" if output_dtype.kind in "iu" else "same_kind"
    if not np.can_cast(data_BAE.dtype, output_dtype, casting=casting):
        raise ValueError(
            f"Cannot convert BTR record from {data_BAE.dtype} "
            f"to {output_dtype} using {casting} casting")

    if check_finite and not np.all(np.isfinite(data_BAE)):
        raise ValueError("BTR record values must be finite")

    data_out = data_BAE.transpose(2, 0, 1).astype(output_dtype, copy=False)

    if check_finite and not np.all(np.isfinite(data_out)):
        raise ValueError("BTR record values must remain finite after conversion")

    heading = validate_array_heading_deg(array_heading_deg)
    np.asarray(heading, dtype=np.float32).tofile(header["fid"])
    data_out.tofile(header["fid"])

def btr_read_sequence(header,seqs_to_read,idx_elv=0):
    """Return ``(data_EBAT, headings_T)`` for selected records/elevations."""

    seqs_to_read = validate_read_indices(seqs_to_read, header["Nseq"], "seqs_to_read")
    idx_elv = validate_read_indices(idx_elv, header["Nelev"], "idx_elv")

    fid = header["fid"]
    Nbands = header["Nbands"]
    Nazim = header["Nazim"]
    Nelev = header["Nelev"]
    npts = len(seqs_to_read)
    dtype = header["dtype"]

    file_dtype = header["dtype"].newbyteorder(header["endian"])

    data_byte_size = dtype_to_bytes(dtype)

    bytes_per_e = Nazim * Nbands * data_byte_size

    data_EBAT = np.empty((len(idx_elv),Nbands,Nazim,npts),dtype=dtype)
    headings_T = np.empty(npts, dtype=np.float32)
    data_bytes_per_record = Nelev * Nazim * Nbands * data_byte_size
    bytes_per_record = 4 + data_bytes_per_record

    for idx,seq in enumerate(seqs_to_read):

        # move the pointer to the right location to read
        seq = int(seq) # force it to python int which can grow without wrapping
        byte_start_seq = header["byte_header"] + seq*bytes_per_record
        fid.seek(byte_start_seq, 0)
        heading = np.fromfile(fid, dtype=header["endian"] + "f4", count=1)
        
        if heading.size != 1:
            raise ValueError(f"Truncated BTR heading at sequence {seq}")
        
        headings_T[idx] = validate_array_heading_deg(heading.item())

        for idxe,ee in enumerate(idx_elv):

            ee = int(ee)
            elv_byte_start = byte_start_seq + 4 + ee*bytes_per_e
            fid.seek(elv_byte_start,0)
            
            temp = np.fromfile(fid,dtype=file_dtype,count=Nazim*Nbands)

            # print(idx)
            # print(temp.shape)
            
            data_EBAT[idxe,:,:,idx] = temp.reshape((Nbands,Nazim))

    return data_EBAT, headings_T

def btr_overwrite_Nseq(header,Nseq):
    Nseq = validate_header_count(Nseq, "Nseq", minimum=0)
    header["fid"].seek(4+4+4,0)
    np.asarray(Nseq, dtype=np.float32).tofile(header["fid"])
