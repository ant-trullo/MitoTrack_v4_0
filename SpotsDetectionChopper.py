"""This function performs 3D spots detection on a time series.

In order to avoid memory errors (very long time series) this function chops the
stack and works on pieces. Results are concatenated in order to have a single
output relative to the entire input matrix.
"""

from importlib import reload
import multiprocessing
import numpy as np

import SpotsDetection3DMultiCore


class SpotsDetectionChopper:
    """Main class, does all the job"""
    def __init__(self, green4D, spots_thr_var, volume_thr_var):

        reload(SpotsDetection3DMultiCore)

        steps    =  green4D.shape[0]
        cpu_owe  =  multiprocessing.cpu_count() - 4
        chop     =  cpu_owe * 3

        if steps > chop:
            n_chops       =  steps // chop
            spots_ints    =  np.zeros((0, green4D.shape[2], green4D.shape[3]), dtype=np.uint16)
            spots_vol     =  np.zeros((0, green4D.shape[2], green4D.shape[3]), dtype=np.int16)
#             spots_lbls  =  np.zeros((0, green4D.shape[1], green4D.shape[2], green4D.shape[3]), dtype=np.int)
            spots_coords  =  np.zeros((0, 4), dtype=np.int16)
            spots_tzxy    =  np.zeros((0, 4), dtype=np.int16)

            for nc in range(n_chops):
                print(nc)
                spots_3D      =  SpotsDetection3DMultiCore.SpotsDetection3DMultiCore(green4D[chop * nc:chop * (nc + 1), :, :, :], spots_thr_var, volume_thr_var)
                spots_ints    =  np.concatenate([spots_ints, spots_3D.spots_ints], axis=0)
                spots_vol     =  np.concatenate([spots_vol, spots_3D.spots_vol], axis=0)
                spots_coords  =  np.concatenate([spots_coords, spots_3D.spots_coords + np.array([nc * chop, 0, 0, 0], np.int16)], axis=0)
                spots_tzxy    =  np.concatenate([spots_tzxy, spots_3D.spots_tzxy + np.array([nc * chop, 0, 0, 0], np.int16)], axis=0)

            spots_3D      =  SpotsDetection3DMultiCore.SpotsDetection3DMultiCore(green4D[chop * (nc + 1):, :, :], spots_thr_var, volume_thr_var)
            spots_ints    =  np.concatenate([spots_ints, spots_3D.spots_ints], axis=0)
            spots_vol     =  np.concatenate([spots_vol, spots_3D.spots_vol], axis=0)
            spots_coords  =  np.concatenate([spots_coords, spots_3D.spots_coords + np.array([(nc + 1) * chop, 0, 0, 0], np.int16)], axis=0)
            spots_tzxy    =  np.concatenate([spots_tzxy, spots_3D.spots_tzxy + np.array([(nc + 1) * chop, 0, 0, 0], np.int16)], axis=0)

        else:
            spots_3D      =  SpotsDetection3DMultiCore.SpotsDetection3DMultiCore(green4D, spots_thr_var, volume_thr_var)
            spots_ints    =  spots_3D.spots_ints
            spots_vol     =  spots_3D.spots_vol
#             spots_lbls  =  spots_3D.spots_lbls
            spots_coords  =  spots_3D.spots_coords
            spots_tzxy    =  spots_3D.spots_tzxy

        self.spots_ints    =  spots_ints
        self.spots_vol     =  spots_vol
        self.spots_tzxy    =  spots_tzxy
#         self.spots_lbls  =  spots_lbls
        self.spots_coords  =  spots_coords


