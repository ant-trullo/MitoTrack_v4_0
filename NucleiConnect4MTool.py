"""This function tracks one single nucleus: it is used in the mitosis tool.

Inputs are the matrix-image of the nuclei to connect, the index of the
single nucleus to track and the distance threshold. This function is used
from a certain time frame (the current frame in the manual tool) up to the end.
"""


import numpy as np
from skimage.measure import regionprops

import NucleiConnectSingle


class NucleiConnect4MTool:
    def __init__(self, left4manual, idx_ref, dist_thr):

        t_tot  =  left4manual.shape[0]                                                    # total number of frames
        ctrs   =  np.zeros((t_tot, 2, left4manual.max().astype(np.int)))                  # initialization of the matrix of the centroids of all the nuclei

        for tt in range(t_tot):
            rgp  =  regionprops(left4manual[tt, :, :].astype(np.int))                     # measuring the centroids positions
            for j in range(len(rgp)):
                ctrs[tt, :, j]  =  rgp[j]['Centroid']

        rgp_ref   =  regionprops((left4manual[0, :, :] == idx_ref).astype(np.int))                            # regionprops of the nucleus to track
        ctrs_ref  =  rgp_ref[0]['centroid']                                                                   # centroid position of the nucleus to track in the first frame
        idx2      =  np.where(ctrs[0, 0, :] == ctrs_ref[0])[0][0]                                             # index of the nucleus to track in the matrix-image
        nucleus   =  NucleiConnectSingle.NucleiConnectSingle(ctrs, left4manual, idx2, 0, dist_thr).labbs2     # tracking of the nucleus     

        self.nucleus  =  nucleus
