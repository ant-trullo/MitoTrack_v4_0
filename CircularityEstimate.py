"""This function calculates the circularity.

It is defined as the ratio between 4*np.pi*A and the p**2 (A is the surface and
p is the perimeter) of all the connected objects present in the image-matrix
labels. This function returns a 2D matrix with the value of the circularity (ies)
and the label(s) of the connected object.
"""


import numpy as np
from skimage import measure


class CircularityEstimate:
    def __init__(self, labels):

        rgp_left   =  measure.regionprops(labels)                 # regionprops of all the connected components of the image-matrix
        circ       =  np.zeros((2, len(rgp_left)))                     # initialization of the matrix to store information

        for k in range(len(rgp_left)):
            circ[:, k]  =  4 * np.pi * rgp_left[k]['Area'] / rgp_left[k]['Perimeter'] ** 2, rgp_left[k]['Label']      # first component circularity, second component label

        self.circ  =  circ
