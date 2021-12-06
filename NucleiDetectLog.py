"""This function  thresholds raw data to deetct nuclei.

It uses the Ostu thresholding algorithm but on the logarithm of the raw data: nuclei
are not homogeneusly bright, the logarithm helps in making distinction with the
the background, not inside the nucleus.
"""

import numpy as np
from skimage import filters
from skimage.morphology import label
from skimage.morphology import remove_small_objects
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import binary_erosion

from PyQt5 import QtWidgets


class NucleiDetectLog:
    """The only class, does all the job"""
    def __init__(self, nucs, otsu_coeff):

        nucs_lbl     =  np.zeros(nucs.shape, dtype=np.int16)
        nucs_fin     =  np.zeros(nucs.shape, dtype=np.bool)
        nucs_th      =  np.zeros(nucs.shape, dtype=np.bool)
        nucs_fsm     =  np.zeros(nucs.shape, dtype=np.int16)
        nucs_log     =  np.log(nucs)
        steps, xlen  =  nucs.shape[:2]

        block_size  =  xlen / 5                                         # for adaptive thresholding
        if block_size % 2 == 0:
            block_size  +=  1


        for t in range(steps):
            val               =  filters.threshold_local(nucs_log[t, :, :], block_size)             # otsu
            nucs_th[t, :, :]  =  nucs_log[t, :, :] > val * otsu_coeff                               # thresholding

        pbar  =  ProgressBar(total1=steps)
        pbar.update_progressbar(0)
        pbar.show()

        for t in range(steps):
            pbar.update_progressbar(t)
            nucs_lbl[t, :, :]  =  label(nucs_th[t, :, :])
            nucs_lbl[t, :, :]  =  remove_small_objects(nucs_lbl[t, :, :].astype(np.int), 5)
            nucs_lbl[t, :, :]  =  binary_fill_holes(nucs_lbl[t, :, :].astype(np.int))
            nucs_fin[t, :, :]  =  binary_erosion(nucs_lbl[t, :, :], iterations=3)                   # from here on is just to smooth nuclei borders
            nucs_fin[t, :, :]  =  binary_dilation(nucs_fin[t, :, :], iterations=4)
            nucs_fsm[t, :, :]  =  filters.median(nucs_fin[t, :, :], np.ones((7, 7)))
            nucs_fsm[t, :, :]  =  label(nucs_fsm[t, :, :], connectivity=1)

        pbar.close()

        self.labbs     =  nucs_fsm



class ProgressBar(QtWidgets.QWidget):
    """Simple progress bar widget"""
    def __init__(self, parent=None, total1=20):
        super(ProgressBar, self).__init__(parent)
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
