"""This function loads .lsm files.

Given the filename of a .lsm files, this function gives as output the matrices
of the red and green channels maximum intensity projected plus the green channel
as it is. Inputs are the file-name and the channel number for nuclei and spots.
"""


import numpy as np
import czifile
from PyQt5 import QtWidgets


class LoadCzi5D:
    """Only class, does all the job"""
    def __init__(self, fname, nucs_spts_ch):

        file_array                 =  np.squeeze(czifile.imread(fname))

        if len(file_array.shape)  == 5:                                                             # case you have more than a time frame
            c, steps, z, x_len, y_len  =  file_array.shape

            pbar  =  ProgressBar(total1=steps)
            pbar.show()

            red_mtx    =  np.zeros((steps, x_len, y_len), dtype='uint16')
            green_mtx  =  np.zeros((steps, x_len, y_len), dtype='uint16')

            for t in range(steps):                                                                  # maximum intensity projection
                pbar.update_progressbar(t)
                for x in range(x_len):
                    red_mtx[t, x, :]    =  file_array[nucs_spts_ch[0], t, :, x, :].max(0)
                    green_mtx[t, x, :]  =  file_array[nucs_spts_ch[1], t, :, x, :].max(0)

            pbar.close()

            self.green4D  =  file_array[0, :, :, :, :]

        else:                                                                                       # case you have just one time frame
            c, z, x_len, y_len  =  file_array.shape

            red_mtx    =  np.zeros((x_len, y_len), dtype='uint16')
            green_mtx  =  np.zeros((x_len, y_len), dtype='uint16')

            for x in range(x_len):                                                                  # maximum intensity projection
                red_mtx[x, :]    =  file_array[nucs_spts_ch[0], :, x, :].max(0)
                green_mtx[x, :]  =  file_array[nucs_spts_ch[1], :, x, :].max(0)

            self.green4D    =  file_array[nucs_spts_ch[1], :, :, :]

        self.red_mtx    =  red_mtx
        self.green_mtx  =  green_mtx


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


