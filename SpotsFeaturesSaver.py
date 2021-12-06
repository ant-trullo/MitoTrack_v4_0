"""This function writes info spots matrix in a compact way.

Instaed of the whole matrix (NxNxN), is saves a matrix (4XN), series
of coordinates and values of all the non zero pixels. This is convenient only in case
of sparse matrix, as it is for spots. For nuclei is not convenient
"""


import numpy as np
from skimage.measure import regionprops_table


class SpotsFeaturesSaver:
    """The only class, does all the job"""
    def __init__(self, spts_mtx, file_name):

        if spts_mtx.sum() > 0:                                                                                                              # check the matrix is not empty
            coords               =  regionprops_table(np.sign(spts_mtx) * 1, properties=["coords"])["coords"][0].astype(np.uint16)          # regionprops of all the spots just to have the coordinates of non zero pixels
            coords_val           =  np.zeros((coords.shape[0] + 1, coords.shape[1] + 1), dtype=np.uint16)                                   # array of positions and value of non zeros pixels
            coords_val[:-1, :3]  =  coords                                                                                                  # add info on the size of the matrix
            coords_val[:-1, 3]   =  spts_mtx[coords[:, 0], coords[:, 1], coords[:, 2]]                                                      # fill with coordinates
            coords_val[-1, :]    =  np.array([spts_mtx.shape[0], spts_mtx.shape[2], spts_mtx.shape[2], 0]).astype(np.uint16)                # fill with pixel values
        else:
            coords_val  =  np.array([spts_mtx.shape[0], spts_mtx.shape[2], spts_mtx.shape[2], 0]).astype(np.uint16)                         # if the matrix is empty, the output will contain only the matrix shape

        np.save(file_name, coords_val)

