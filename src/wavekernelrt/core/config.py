import numpy as np
# Create a class to carry around constants that will get used in more than one place and are a design choice by the user.

class Config:
    # for curved cylinders, compute the cosine between the incoming angles and array
    # anything below the cutoff below is turned off. All negative values mean the elements
    # are facing away from the incoming wave (baffled). setting it to cos(60*pi/180) would
    # shut off all backside angles and angles coming in less than 60 degrees (conserative) and
    # narrows the aperture
    CURVATURE_SHUTOFF=np.cos(60*np.pi/180)

    PHS_PREC_WARN_DEG=0.5
    PHS_PREC_ERR_DEG=5