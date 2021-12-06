"""This function loads info spots as matrix from their compact way.

From a saved (4XN) matrix ()coordinates and values of all the non zero pixels)
this function rebuilts the 3D matrix
"""


import numpy as np


class SpotsFeaturesLoader:
    """The only class, does all the job"""
    def __init__(self, file_name):

        coords_val  =  np.load(file_name)                                                                       # load the data
        if coords_val.size > 4:
            spts_mtx    =  np.zeros((coords_val[-1, 0], coords_val[-1, 1], coords_val[-1, 2]), np.uint16)           # define the output size from data

            spts_mtx[coords_val[:-1, 0], coords_val[:-1, 1], coords_val[:-1, 2]]  =  coords_val[:-1, 3]             # insert output data

            self.spts_mtx  =  spts_mtx

        else:
            self.spts_mtx  =  np.zeros((coords_val[0], coords_val[1], coords_val[2]), np.uint16)

