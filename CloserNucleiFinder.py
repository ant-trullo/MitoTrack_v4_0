"""This function finds the connected object closer to the starting point.

Input are 'nuclei_lbl', which is the matrix-image with
connected labeled objects, points, the coordinates of the point and spt_nuc_maxdist,
the maximum distance from the point to the nucleus. It returns the label value 
of the closest object. To do that, it starts from a square centred in point and
with a side of 1.
If the maximum in the square is zero, the side is increased and the maximum
is searched into a broader square.
"""

import numpy as np


class CloserNucleiFinder:
    def __init__(self, nuclei_lbl, point, spt_nuc_maxdist):

        point            =  np.round(point).astype(np.int)      # coordinates must be integer
        mx_pt            =  0                                   # initialization of the output        
        side             =  1                                   # initialization of the edge
        edge_x, edge_y   =  nuclei_lbl.shape                    # image size, to be sure the algorithm does not search out of the image

        while mx_pt == 0 and side < spt_nuc_maxdist:
            mx_pt  =   (nuclei_lbl[np.max((point[0] - side, 0)):np.min((point[0] + side, edge_x - 1)) + 1, np.max((point[1] - side, 0)):np.min((point[1] + side, edge_y - 1)) + 1]).max()     # the maximum is searched taken into account for the image borders
            side   +=  1

        self.mx_pt  =  mx_pt
