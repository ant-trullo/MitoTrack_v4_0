"""Given tracked spots and tracked nuclei, this function generates the false colored video.

All nuclei here have tag one when inactive, tag 2 when active.


NucleiSpotsConnection4Merged does the same job but it works on the final matrix and reassociate
nuclei and spots.
"""

import numpy as np
from skimage.measure import label
from skimage.measure import regionprops, regionprops_table
from PyQt5 import QtWidgets

import CloserNucleiFinder


class NucleiSpotsConnection:
    """Class to connect spots and nuclei during the cycle"""
    def __init__(self, spots_tracked, nuclei_tracked):

        nuclei_active  =  np.sign(nuclei_tracked).astype(np.uint8)
        idx            =  spots_tracked[spots_tracked != 0]                                  # list of all the values in the spots matrix. The zero, which is the first, is removed
        idx            =  np.unique(idx)


        pbar  =  ProgressBar(total1=idx.size)
        pbar.show()

        i         =  0

        for k in idx:
            pbar.update_progressbar(i)
            aa       =  (spots_tracked == k).sum(2).sum(1)                      # spot by spot we extract the time series of the frame in which the spot is present (the nucleu is active)
            t_steps  =  np.where(aa != 0)[0]
            i  +=  1

            for tt in t_steps:
                nuclei_active[tt, :, :]  +=  (nuclei_tracked[tt, :, :] == k)            # when nucleus is active we sum it to its calc (read we give it tag 2)

        pbar.close()

        nuclei_active3c              =   np.zeros((nuclei_tracked.shape[0], nuclei_tracked.shape[1], nuclei_tracked.shape[2], 3), dtype=np.uint8)    # three channel version of the matrix to have colors (blue -inactive nuclei, red - active nuclei, green -spots)
        nuclei_active3c[:, :, :, 0]  =   (nuclei_active == 2)
        nuclei_active3c[:, :, :, 2]  =   (nuclei_active == 1)
        nuclei_active3c[:, :, :, 1]  =   np.sign(spots_tracked)
        nuclei_active3c              *=  255

        n_active_vector  =  np.zeros(nuclei_active.shape[0])
        for t in range(n_active_vector.size):                                                                           # vector of the activation: how many nuclei are active in each time frame
            l1                  =  label((nuclei_active[t, :, :] == 2) * nuclei_tracked[t, :, :], connectivity=1)
            n_active_vector[t]  =  l1.max()


        self.nuclei_active    =  nuclei_active
        self.nuclei_active3c  =  nuclei_active3c
        self.n_active_vector  =  n_active_vector




class NucleiSpotsConnection4Merged:
    """Class to connect spots and nuclei along two cycles (partial merged)"""
    def __init__(self, nuclei_segm, spots_tracked, max_dist_ns):


        nuclei_active  =  np.sign(nuclei_segm).astype(np.uint8)
#         idx            =  np.unique(spots_tracked)[1:]                                                             # list of all the values taken by the spots. The zero, which is the first, is removed
        rgp_spts_trk  =  regionprops_table(spots_tracked, properties=["label", "coords"])

        pbar  =  ProgressBar(total1=rgp_spts_trk["label"].size)
        pbar.show()

#         i         =  0

        for p_idx, k in enumerate(rgp_spts_trk["coords"]):
            pbar.update_progressbar(p_idx)
            spot_trk                             =  np.zeros(spots_tracked.shape, dtype=np.uint8)
            spot_trk[k[:, 0], k[:, 1], k[:, 2]]  =  1
#             aa       =  (spots_tracked == k).sum(2).sum(1)
            t_steps                              =  np.where(spot_trk.sum(2).sum(1) != 0)[0]
#             i  +=  1

            for tt in t_steps:
                sp_lbls  =  label(spot_trk[tt])
                mxs      =  np.array([])
                rgp      =  regionprops(sp_lbls)

                for jj in range(len(rgp)):
                    sss  =  CloserNucleiFinder.CloserNucleiFinder(nuclei_segm[tt], np.array([rgp[jj]['centroid'][0], rgp[jj]['centroid'][1]]), max_dist_ns).mx_pt     # find the nucleus closer to the spot
                    if sss != 0:
                        mxs  =  np.append(mxs, sss)

                mxs  =  np.unique(mxs)
                for x in mxs:
                    nuclei_active[tt]  +=  (nuclei_segm[tt] == x).astype(np.uint8)                        # changes the tag of the active spot; from 1 to 2

        pbar.close()

        nuclei_active3c              =   np.zeros((nuclei_segm.shape[0], nuclei_segm.shape[1], nuclei_segm.shape[2], 3), dtype=np.uint8)    # three channel version of the matrix to have colors (blue -inactive nuclei, red - active nuclei, green -spots)
        nuclei_active3c[:, :, :, 0]  =   (nuclei_active == 2)
        nuclei_active3c[:, :, :, 2]  =   (nuclei_active == 1)
        nuclei_active3c[:, :, :, 1]  =   np.sign(spots_tracked)
        nuclei_active3c              *=  255

        n_active_vector  =  np.zeros(nuclei_active[:, 0, 0].size)
        for t in range(n_active_vector.size):                                                                                   # vector of the activation: how many nuclei are active in each time frame
            l1                  =  label((nuclei_active[t, :, :] == 2) * nuclei_segm[t, :, :], connectivity=1)
            n_active_vector[t]  =  l1.max()


        self.nuclei_active    =  nuclei_active
        self.nuclei_active3c  =  nuclei_active3c
        self.n_active_vector  =  n_active_vector



class ProgressBar(QtWidgets.QWidget):
    """Simple progressbar widget"""
    def __init__(self, parent=None, total1=20):
        super(ProgressBar, self).__init__(parent)
        self.name_line1  =  QtWidgets.QLineEdit()

        self.progressbar  =  QtWidgets.QProgressBar()
        self.progressbar.setMinimum(1)
        self.progressbar.setMaximum(total1)


        main_layout  =  QtWidgets.QGridLayout()
        main_layout.addWidget(self.progressbar, 0, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Progress")
        self.setGeometry(500, 300, 300, 50)


    def update_progressbar(self, val1):
        """Progressbar updater"""
        self.progressbar.setValue(val1)
        QtWidgets.qApp.processEvents()


