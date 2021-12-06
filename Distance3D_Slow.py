"""This function calculate the distances between a point p_start and a set of other points.

Only distances with a distance smaller than dist_thr are consindered.
"""


import numpy as np


class Distance3D_Slow:
    """The only class, does all the job"""
    def __init__(self, pt_start, ctrs, dist_thr):

        dis_size  =  ctrs.shape[0]
        dists     = np.zeros([ctrs.shape[0]], dtype=np.int)

        for k in range(dis_size):
            dists[k]  =  np.sum((pt_start[0] - ctrs[k, 1]) ** 2 + (pt_start[1] - ctrs[k, 2]) ** 2 + (pt_start[2] - ctrs[k, 3]) ** 2)   # square of the euclidean 3D distances between pt_start and a slice of ctrs

        if dists.min() < dist_thr ** 2:
            idx  =  np.argmin(dists)
        else:
            idx  =  -1

        self.idx  =  idx

