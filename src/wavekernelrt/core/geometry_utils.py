import numpy as np

# helper function to compute the distance between adjacent points down the N dimension in M dimensional space
def segment_lengths(xyz_NM):
    dxyz = np.diff(xyz_NM,axis=0)
    return np.linalg.norm(dxyz, axis=1)