"""This function loads .lsm or .tif data file (just one).

Given the filename as input, the output are the matrices of the maximum intensity
projection of red and green channels (nuclei and spots respectively).
"""

import numpy as np
import tifffile
from PyQt5 import QtGui, QtWidgets


class LoadLsmOrTif5D:
    def __init__(self, fname, nucs_spts_ch):

        file_array                 =  tifffile.imread(fname)                    # load file
        steps, z, c, x_len, y_len  =  file_array.shape                          # shape info

        pbar  =  ProgressBar(total1=steps)
        pbar.show()

        red_mtx    =  np.zeros((steps, x_len, y_len))                           # initialization of nuclei matrix
        green_mtx  =  np.zeros((steps, x_len, y_len))                           # initialization of spots matrix

        for t in range(steps):
            pbar.update_progressbar(t)
            for x in range(x_len):
                red_mtx[t, x, :]    =  file_array[t, :, nucs_spts_ch[0], x, :].max(0)         # maximum intensity projection
                green_mtx[t, x, :]  =  file_array[t, :, nucs_spts_ch[1], x, :].max(0)         # maximum intensity projection


        self.red_mtx    =  red_mtx
        self.green_mtx  =  green_mtx


class ProgressBar(QtGui.QWidget):
    def __init__(self, parent=None, total1=20):
        super(ProgressBar, self).__init__(parent)
        self.name_line1  =  QtGui.QLineEdit()

        self.progressbar  =  QtWidgets.QProgressBar()
        self.progressbar.setMinimum(1)
        self.progressbar.setMaximum(total1)


        main_layout  =  QtGui.QGridLayout()
        main_layout.addWidget(self.progressbar, 0, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Progress")
        self.setGeometry(500, 300, 300, 50)


    def update_progressbar(self, val1):
        self.progressbar.setValue(val1)
        QtWidgets.qApp.processEvents()
