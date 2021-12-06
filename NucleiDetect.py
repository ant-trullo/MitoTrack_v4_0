"""This function detects nuclei:

Starting from the Maximum Intensity Projection of the nuclei raw data,
this function detect all the nuclei calcs, giving as a result a B&W
image series, blach for the background and white for nuclei.
"""

import numpy as np
from skimage.morphology import label, remove_small_objects
from skimage import filters
from scipy import ndimage
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import binary_erosion
from PyQt5 import QtWidgets



class NucleiDetect:
    """The only class, does all the job"""
    def __init__(self, nucleiMatx, gauss_filter_size):

        nucleiMatxSM  =  np.zeros(nucleiMatx.shape)
        steps         =  nucleiMatx.shape[0]

        pbar  =  ProgressBar(total1=steps * 3)
        pbar.show()

        for t in range(steps):
            pbar.update_progressbar(t)
            nucleiMatxSM[t, :, :]  =  filters.gaussian(nucleiMatx[t, :, :], gauss_filter_size)    # pre-smoothing filter

        im_seg  =  np.zeros(nucleiMatxSM.shape, np.bool)

        for t in range(steps):
            pbar.update_progressbar(t + steps)
            val              =  filters.threshold_otsu(nucleiMatxSM[t, :, :])                  # threshold us calculated frame by frame by Otsu algorithm
            im_seg[t, :, :]  =  nucleiMatxSM[t, :, :] > val                                    # thresholding, frame by frame

#         im_segO    =  np.zeros(im_seg.shape, dtype=np.bool)
#         im_segOC   =  np.zeros(im_seg.shape, dtype=np.bool)
#         labbs      =  np.zeros(im_seg.shape, dtype=np.int16)
# 
#         for k in range(im_segO[:, 0, 0].size):
#             pbar.update_progressbar(k + 2 * steps)
#             im_segO[k, :, :]   =  ndimage.morphology.binary_opening(im_seg[k, :, :])           # the segmented result is cleaned by opening and closing binary operations
#             im_segOC[k, :, :]  =  ndimage.morphology.binary_closing(im_segO[k, :, :])
#             labbs[k, :, :]     =  label(im_segOC[k, :, :], connectivity=1)              # final pre-labeling to prepare result for the segmentantion

        nucs_lbl     =  np.zeros(im_seg.shape, dtype=np.int16)
        nucs_fin     =  np.zeros(im_seg.shape, dtype=np.bool)
#         nucs_th      =  np.zeros(nucs.shape, dtype=np.bool)
        nucs_fsm     =  np.zeros(im_seg.shape, dtype=np.int16)

        for t in range(steps):
            pbar.update_progressbar(t + 2 * steps)
            nucs_lbl[t]  =  label(im_seg[t])
            nucs_lbl[t]  =  remove_small_objects(nucs_lbl[t].astype(np.int), 5)
            nucs_lbl[t]  =  binary_fill_holes(nucs_lbl[t].astype(np.int))
            nucs_fin[t]  =  binary_erosion(nucs_lbl[t], iterations=3)                   # from here on is just to smooth nuclei borders
            nucs_fin[t]  =  binary_dilation(nucs_fin[t, :, :], iterations=4)
            nucs_fsm[t]  =  filters.median(nucs_fin[t], np.ones((7, 7)))
            nucs_fsm[t]  =  label(nucs_fsm[t], connectivity=1)

        pbar.close()

        self.labbs  =  nucs_fsm


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
