"""This function tracks already segmented spots.

Given the matrix of the segmented spots and their
coordinates this function tracks in 3D all the spots present in the
first frame following a minimal distance criteria.
"""


import numpy as np
from PyQt5 import QtWidgets
from importlib import reload

# import Distance3D_Utility
import Distance3D_Slow


class SpotTracker:
    """Coordinate the single spot tracking to track all the spots in the matrix"""
    def __init__(self, spots_tzxy, dist_thr, t_track_end):

        steps  =  spots_tzxy[:, 0].max() + 1

        spts_trck_info  =  []

        if t_track_end == 0:
            frames2work  =  [0]
        else:
            frames2work  =  range(t_track_end + 1)

        pbar  =  ProgressBar(total1=len(frames2work))
        pbar.show()
        idx_pbar  =  0

        for ttt in frames2work:
            pbar.update_progressbar(idx_pbar)
            idx_pbar  +=  1

            spots_tzxy_t0  =  spots_tzxy[spots_tzxy[:, 0] == ttt]                                                     # estracting the sub-matrix from coord matrix with time 0 to start
            for tag in range(spots_tzxy_t0.shape[0]):                                                                 # tag by tag
                pt_start  =  list(spots_tzxy_t0[tag, 1:])                                                             # starting point of the spot from the coord matrix
                spts_trck_info.append(SingleSpotTracker(pt_start, ttt, steps, spots_tzxy, dist_thr).spots_trkd)       # tracking of the single spot to add to the output matrix

        pbar.close()
        self.spts_trck_info  =  spts_trck_info


class SingleSpotTracker:
    """Single spot tracking"""
    def __init__(self, pt_start, t, steps, spots_tzxy, dist_thr):

        reload(Distance3D_Slow)
        spots_trkd  =  np.array([t, pt_start[0], pt_start[1], pt_start[2]])

        for tt in range(t, steps - 1):
            ctrs_est  =  spots_tzxy[spots_tzxy[:, 0] == tt + 1].astype(np.int)                              # extract the sub-matrix of the coordinate having as time the and step
            if ctrs_est.shape[0] > 0:
                # idx       =  Distance3D_Utility.Distance3D(pt_start, ctrs_est, dist_thr ** 2)
                idx       =  Distance3D_Slow.Distance3D_Slow(pt_start, ctrs_est, dist_thr ** 2).idx
                if idx >= 0:                                                                     # if distance is not -1, a spots was found
                    pt_start                =  list(ctrs_est[idx, 1:])                                          # update the starting position for the next frame
                    idx_map                 =  np.where((spots_tzxy == ctrs_est[idx, :]).all(axis=1))[0][0]      # map the index from sub-coord-matrix to coord-matrix
                    spots_trkd              =  np.concatenate((spots_trkd, spots_tzxy[idx_map, :]), axis=0)
                    spots_tzxy[idx_map, :]  =  0                                                                  # remove the and and the coord-matrix to avoid and association
                else:
                    break

        self.spots_trkd    =  spots_trkd.reshape((spots_trkd.size // 4, 4))


# class MipMinusTracked:
#    """Remove the tracked spots (in 3D) from the 4D raw data and redo mip"""
#    def __init__(self, green4D, spots_trkd, t_start_value):

#        steps, x_len          =  green4D.shape[0], green4D.shape[2]                                    # number of time steps and x size (for mip)
#        green4D              *=  1 - np.sign(spots_trkd)                                               # removing tracked spots from the matrix
#        imarray_green_clean   =  np.zeros((green4D.shape[0], green4D.shape[2], green4D.shape[3]))      # initializing the output matrix

#        for t in range(steps):                                                                  # maximum intensity projection
#            for x in range(x_len):
#                imarray_green_clean[t, x, :]  =  green4D[t, :, x, :].max(0)

#        self.imarray_green_clean  =  imarray_green_clean
#        self.green4D              =  green4D


class ProgressBar(QtWidgets.QWidget):
    """Simple progress bar widget"""
    def __init__(self, parent=None, total1=20):
        super().__init__(parent)
        self.name_line1  =  QtWidgets.QLineEdit()

        self.progressbar1  =  QtWidgets.QProgressBar()
        self.progressbar1.setMinimum(1)
        self.progressbar1.setMaximum(total1)

        main_layout  =  QtWidgets.QGridLayout()
        main_layout.addWidget(self.progressbar1, 0, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Progress")
        self.setGeometry(500, 300, 300, 50)

    def update_progressbar(self, val1):
        """Progress bar updater"""
        self.progressbar1.setValue(val1)
        QtWidgets.qApp.processEvents()
