"""This function tracks a single nucleus in a time series.

Inputs are the matrix with all the centroids of all the nuclei,
the time series of the segmented nuclei, the index of the nucleus
to track, the starting frame to start the tracking and the
distance threshold.
"""



import numpy as np

import DistancesP2Vec


class NucleiConnectSingle:
    """The only class, does all the job"""
    def __init__(self, ctrs, labbs, idx_ref, t1, dist_thr):

        ctrs    =  np.round(ctrs).astype(np.int)                                                                                                                # the matrix with the centroids of the nuclei must be integer to work as a coordinate for a matrix 
        labbs2  =  np.zeros(labbs.shape, dtype=np.uint16)                                                                                                        # initialization of the output matrix
        t_tot   =  labbs.shape[0]                                                                                                                               # number of time frames

        if ctrs[t1, 0, idx_ref] != 0:                                                                                                                           # check the centroid is there (just to prevent errors)

            labbs2[t1, :, :]  +=  (labbs[t1, :, :] == labbs[t1, ctrs[t1, 0, idx_ref], ctrs[t1, 1, idx_ref]])                                                    # the nucleus is initialized in the first frame

            for t in range(t1, t_tot - 1):
                dist                 =  DistancesP2Vec.DistancesP2Vec(ctrs[t, :, idx_ref], ctrs[t + 1, :, :])                                                   # measure of the distances of the centroid of the given nucleus and the centroids of all the other nuclei in the following frame 
                ctrs[t, :, idx_ref]  =  np.array([0, 0])                                                                                                        # the coordinate of the centroid of the nucleus are removed when the nucleus is included in the final matrix

                if dist.dists_sqrt.min() < dist_thr and labbs[t + 1, ctrs[t + 1, 0, dist.dists_sqrt.argmin()], ctrs[t + 1, 1, dist.dists_sqrt.argmin()]] > 0:    # check the nucleus with closer centroid and check the centroid is in the calc of the nucleus itself
                    idx_ref              =   dist.dists_sqrt.argmin()
                    labbs2[t + 1, :, :]  +=  (labbs[t + 1, :, :] == labbs[t + 1, ctrs[t + 1, 0, idx_ref], ctrs[t + 1, 1, idx_ref]])                              # if requirements are fulfilled, the new nucleus is added to the final matrix with its proper tag, otherwise is left

                else:
                    break

        self.labbs2  =  labbs2
        self.ctrs    =  ctrs


