import numpy as np

def sio_open(filename, sample_rate=None):

    header = dict()
    header["fid"] = open(filename,"rb")

    header["id"] = int(np.fromfile(header["fid"],dtype=">i4",count=1)[0])            # ID Number
    header["num_records"] = int(np.fromfile(header["fid"],dtype=">i4",count=1)[0])   # Number of Records in File
    header["seq_len_bytes"] = int(np.fromfile(header["fid"],dtype=">i4",count=1)[0]) # Record Length in Bytes
    header["num_channels"] = int(np.fromfile(header["fid"],dtype=">i4",count=1)[0])  # Number of Channels in File
    header["bytes_per_pt"] = int(np.fromfile(header["fid"],dtype=">i4",count=1)[0])  # Bytes/ Point

    jnk = np.fromfile(header["fid"],dtype=">i4",count=1) # not used; 0 = integer, 1 = real

    header["pts_per_channel"] = int(np.fromfile(header["fid"],dtype=">i4",count=1)[0])  #  Number of Points per Channel
    header["pts_per_seq"] = header["seq_len_bytes"] // header["bytes_per_pt"]
    header["block_size"] = int(np.fromfile(header["fid"],dtype=">i4",count=1)[0])  #  should be 32677

    filename = header["fid"].read(24) # Read in the file name
    comments = header["fid"].read(72) # Read in the Comment String

    header["native_filename"] = filename.decode("ascii",errors="replace").rstrip("\x00").strip()
    header["native_comments"] = comments.decode("ascii",errors="replace").rstrip("\x00").strip()

    header["num_seqs"] =  header["num_records"] // header["num_channels"]
    header["header_bytes"] = 4*8+24+72

    if sample_rate is not None:
        # figure out some derived parameters
        header["secs_per_record"] = (header["seq_len_bytes"] / header["bytes_per_pt"]) / sample_rate
        header["total_time_secs"] = header["num_seqs"] * header["secs_per_record"]
        header["sample_rate"] = sample_rate

    return header

def sio_read(header,seqs_to_read):

    # data is laid out as
    # [header (length seq_len_bytes)][ch 0 seq 0][ch 1 seq 0][... ch N-1 seq 0][ch 0 seq 1]
    # [ch 0 seq 1][ch 1 seq 1][... ch N-1 seq 1]
    seqs_to_read = np.atleast_1d(seqs_to_read).astype(int)

    fid = header["fid"]
    nchan = header["num_channels"]
    pts_per_seq = header["seq_len_bytes"]//header["bytes_per_pt"]
    npts = len(seqs_to_read)*pts_per_seq

    if header["bytes_per_pt"]==2:
        dtype = '>i2'
    elif header["bytes_per_pt"]==4:
        dtype = 'float32'
    else:
        raise ValueError("Unknown data type")

    data_NT = np.empty((nchan,npts),dtype=dtype)

    # remember python is 0 based
    # sequences are assumed to be zero based as well
    for idx,seq in enumerate(seqs_to_read):
            # move the pointer to the right location to read
            byte_start_seq = header["seq_len_bytes"] + seq*header["seq_len_bytes"]*nchan
            fid.seek(byte_start_seq,0)

            # read in the block
            idx1 = idx*pts_per_seq
            idx2 = idx1 + pts_per_seq
            for ch in range(nchan):
                data_NT[ch,idx1:idx2] = np.fromfile(fid,dtype=dtype,count=pts_per_seq)


    return data_NT

def sio_close(header):
    header["fid"].close()

def convert_geometry(file_in,file_out):

    data = np.loadtxt(file_in)

    # read in as meters, but wavekernel expects geometries to be in inches
    # also gives it as north, east, depth
    posx = data[:,2]*3.2808*12
    posy = data[:,1]*3.2808*12
    posz = data[:,3]*3.2808*12

    posx -= np.mean(posx)
    posy -= np.mean(posy)
    posz -= np.mean(posz)

    with open(file_out,"w") as file:
        [file.write(f"{x} {y} {z}\n") for x,y,z in zip(posx,posy,posz)]



