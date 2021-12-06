"""This function calculates the distances between a point and a vector.

Given a point like (x0 = (x0_1, x0_2)) and a vector (vec, such that vec.shape = (2, n))
the output is a vector of euclidean distances of the point from all the component of
the vector.
"""


import numpy as np


class DistancesP2Vec:

    def __init__(self, x0, vec):

        dists       =  np.zeros(vec[0, :].size)         # initialization of the square distance vector
        dists_sqrt  =  np.zeros(vec[0, :].size)         # initialization of the distance vector

        for i in range(dists.size):
            dists[i]       =  (x0[0] - vec[0, i]) ** 2 + (x0[1] - vec[1, i]) ** 2           # the distance is euclidean
            dists_sqrt[i]  =  np.sqrt(dists[i])

        self.dists       =  dists
        self.dists_sqrt  =  dists_sqrt
