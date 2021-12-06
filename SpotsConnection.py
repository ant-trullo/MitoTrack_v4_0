"""This function connects the detected spots to the tracked nuclei.

Input are tracked nuclei, detected spots and the maximum distance between
nuclei and spots.
"""



import numpy as np
from skimage.morphology import label
from skimage.measure import regionprops
from PyQt5 import QtWidgets

import CloserNucleiFinder


class SpotsConnection:
    """The only class, does all the job"""
    def __init__(self, nuclei_tracked, spots_mask, spt_nuc_maxdist):

        spots_tracked   =  np.zeros(spots_mask.shape, dtype=np.int16)               # initialization of the spots tracked matrix
        t_tot           =  spots_mask.shape[0]                                    # total number of time steps
        spots_mask_lbl  =  np.zeros(spots_mask.shape, dtype=np.int16)               # matrix for the random labeling of the spots

        pbar  =  ProgressBar(total1=3 * t_tot)                                    # progressbar initialization
        pbar.show()

        for t in range(t_tot):
            pbar.update_progressbar(t)
            spots_mask_lbl[t, :, :]  =  label(spots_mask[t, :, :]).astype(np.int16)          # spots labeling: nuclei are not involved here, this labeling serves only to identify each spots and work with it

        for t in range(t_tot):
            pbar.update_progressbar(t_tot + t)
            a  =  np.unique(spots_mask_lbl[t, :, :])[1:]                # tags of the spots present in the time frame
            for k in a:
                spots_tracked[t, :, :]  +=  (spots_mask_lbl[t, :, :] == k) * ((spots_mask_lbl[t, :, :] == k) * nuclei_tracked[t, :, :]).max()      # overlapping spot with the nuclei matrix: the value of the nucleus will give the new tag to the spot

        left      =  np.sign(spots_mask) - np.sign(spots_tracked)                                           # if there is no overlapping, the new spot tag will be zero. Here the algorithm recognizes the spots still tag-less
        left_lbl  =  np.zeros(left.shape)
        for t in range(t_tot):
            left_lbl[t, :, :]  =  label(left[t, :, :])

        for tt in range(t_tot):
            pbar.update_progressbar(2 * t_tot + tt)
            rgp  =  regionprops(left_lbl[tt, :, :].astype(np.int))
            for kk in range(len(rgp)):
                m                        =   CloserNucleiFinder.CloserNucleiFinder(nuclei_tracked[tt, :, :], np.array([rgp[kk]["Centroid"][0], rgp[kk]["Centroid"][1]]), spt_nuc_maxdist).mx_pt    # the algorithm searches the nuclei closer to the spot to give the spot a tag
                spots_tracked[tt, :, :]  +=  (left_lbl[tt, :, :] == rgp[kk]['label']).astype(np.int) * m

        pbar.close()

        self.spots_tracked  =  spots_tracked



class ProgressBar(QtWidgets.QWidget):
    """Simple progress bar widget"""
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
        """Progress bar updater"""
        self.progressbar.setValue(val1)
        QtWidgets.qApp.processEvents()
