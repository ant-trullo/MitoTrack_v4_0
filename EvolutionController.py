"""This function checks the tracking.

It works on the whole time series (before, during and after mitosis). For each
label it checks that you have one connected component at the beginning and two
at the end (if one sister escape we don't know if it is the fast activating or
the slow activating one).

The EvolutionControllerDM class works as the EvolutionControllerTOT class,
but on the during mitosis part only. The difference is that in case of
tracking errors, the software automatically tries to fix the tracking error
by lunching the NucleiConnect4MTool function.

"""



import numpy as np
# from skimage.morphology import label   #   , remove_small_objects
from skimage.measure import regionprops_table
from PyQt5 import QtWidgets

import NucleiConnect4MTool
import SpotsConnection


class EvolutionControllerTOT:
    """Checks if there are problems in the tracking of the merged time series.
    In this class, we put the constrains explained above only between the beginning
    of 13th cycle and the first activatio of the second activating daughters. After
    the activation of both sister we don't care if nucleus goes out or is badly
    tracked. If the 2 sisters never activate, we keep the constrain up to the end
    of 14th cycle.
    """
    def __init__(self, conc_nuc, conc_wild, spots_tracked, frames_bm, frames_dm, max_dist_ns):

        steps        =  conc_nuc.shape[0]                                             # number of time steps
        idxs         =  spots_tracked[spots_tracked != 0]                             # trick to speed-up np.unique
        idxs         =  np.unique(idxs)                                               # array with all the labels to work on    

        spots_special  =  SpotsConnection.SpotsConnection(conc_wild[frames_bm + frames_dm - 1:], np.sign(spots_tracked[frames_bm + frames_dm - 1:]), max_dist_ns).spots_tracked   # spots associated to the wild nuclei (we have a different label for spots associated to daughters)
        spots_special  =  np.concatenate((np.zeros((frames_bm + frames_dm - 1, spots_special.shape[1], spots_special.shape[2]), dtype=np.uint16), spots_special), axis=0)          # padding with zeros to obtain the same shape

        pbar   =  ProgressBar(total1=idxs.size)                                       # progress bar initialization
        pbar.show()

        rgp_tracked   =  regionprops_table(spots_tracked, properties=["label", "coords"])
        conc_nuc_ok   =  np.zeros(conc_nuc.shape, dtype=np.uint16)

        for p_idx, k in enumerate(rgp_tracked['coords']):
            pbar.update_progressbar(p_idx)
#             p_idx  +=  1
#             print("p_idx = " + str(p_idx))                        
            spt       =  spots_special[k[:, 0], k[:, 1], k[:, 2]]               # isolate spots of two daugheters keeping for them different labels
            spt_idxs  =  spt[spt != 0]
            spt_idxs  =  np.unique(spt_idxs)
            if spt_idxs.size >= 3:                                          # three labels, there is a problem so t_end = 0 and the tag is automtically discarded
                t_end  =  0
            if spt_idxs.size <= 1:                                          # one label or less: daughters are not both activating. Control will be done up to the end
                t_end  =  steps
            if spt_idxs.size == 2:                                          # two labels: both daughters activates, control up to the activation of the latest activating
#                 print("1")
                t_end0  =  []
                for t in range(k.shape[0]):
                    if spots_special[k[t, 0], k[t, 1], k[t, 2]] == spt_idxs[0]:
                        t_end0.append(k[t, 0])                                      # search the first frame the first daughter is on
                        break

                t_end1  =  []
                for t in range(k.shape[0]):
                    if spots_special[k[t, 0], k[t, 1], k[t, 2]] == spt_idxs[1]:
                        t_end1.append(k[t, 0])                                      # search the first frame the second daughter is on
                        break

                t_end  =  max(t_end0, t_end1)[0]                                    # take the maximum of both
#                 print("2")

            lblmax   =  np.zeros(t_end, np.uint16)
            family   =  (conc_nuc == rgp_tracked["label"][p_idx]) * conc_wild           # isolate mother and its daughters
            for hh in range(lblmax.size):
                bff         =  family[hh]
                lblmax[hh]  =  np.unique(bff[bff != 0]).size

            if lblmax.size > 0:
                if lblmax[0] == 1 and lblmax[-1] == 2 and np.sum(np.abs(np.diff(lblmax))) == 1:                  # given a nucleus label, you have one connected component at the beginning and two at the end: we check, by absolute
#                     print(rgp_tracked["label"][p_idx])
                    conc_nuc_ok  +=  np.uint16(rgp_tracked["label"][p_idx]) * np.sign(family)                    # value of derivative summed, that there is just one jump (from 1 to 2 connected components)
#                 else:
#                     print(rgp_tracked["label"][p_idx])

        conc_nuc_wrong  =  conc_nuc.astype(np.uint16) * (1 - np.sign(conc_nuc_ok))
        pbar.close()

        self.conc_nuc_ok     =  conc_nuc_ok
        self.conc_nuc_wrong  =  conc_nuc_wrong



class EvolutionControllerDM:
    """Checks if there are problems in the tracking during mitosis."""
    def __init__(self, conc_nuc, labbs):

        steps        =  conc_nuc.shape[0]
        idxs         =  conc_nuc[conc_nuc != 0]
        idxs         =  np.unique(idxs)
        conc_nuc_ok  =  np.zeros(conc_nuc.shape, dtype=np.uint16)

        pbar   =  ProgressBar(total1=idxs.size)
        p_idx  =  0
        pbar.show()

        for k in idxs:
            pbar.update_progressbar(p_idx)
            p_idx   +=  1
            lblmax   =  np.zeros(steps)
# #             family   =  np.zeros(conc_nuc_ok.shape[1:])
#             for t in range(steps):
# #                 family     =  remove_small_objects(family.astype(np.uint16), min_size=4)
# #                 family     =  label((conc_nuc[t, :, :] == k).astype(np.uint16), connectivity=1)
#                 family     =  remove_small_objects((conc_nuc[t] == k) * labbs[t], min_size=4)
#                 family     =  label(family, connectivity=1)
#                 lblmax[t]  =  family.max()
# 

            family     =  (conc_nuc == k) * labbs
            for t in range(steps):
#                 family     =  remove_small_objects(family.astype(np.uint16), min_size=4)
                bff        =  family[t]
                lblmax[t]  =  np.unique(bff[bff != 0]).size




            if lblmax[0] == 1 and np.sum(np.abs(np.diff(lblmax))) == 1:
                conc_nuc_ok  +=  np.uint16(k) * (conc_nuc == k)

        conc_nuc_wrong  =  conc_nuc * (1 - np.sign(conc_nuc_ok))
        pbar.close()

        nuc_supp  =  np.zeros(conc_nuc_wrong.shape, dtype=np.uint16)
        idxs      =  np.unique(conc_nuc_wrong[0, :, :])[1:]
        for k in idxs:
            nuc_supp  +=  NucleiConnect4MTool.NucleiConnect4MTool(conc_nuc_wrong, k, 8).nucleus.astype(np.uint16)

        conc_nuc_ok     +=  nuc_supp
        conc_nuc_wrong  *=  1 - np.sign(nuc_supp)

        self.conc_nuc_ok     =  conc_nuc_ok.astype(np.uint16)
        self.conc_nuc_wrong  =  conc_nuc_wrong.astype(np.uint16)



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


