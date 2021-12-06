"""This function takes he same label to mother and daughters during the all process.

The output is given by three matrices: one in which both daughters have the same label of the mother,
one in which all the original labels are kept (useful in other parts of the code) and one with the
concatenate spots.
"""


import multiprocessing
import numpy as np
from skimage.morphology import remove_small_objects, label
from PyQt5 import QtWidgets

import DuringMitosisDaughterLabel
import SpotsFeaturesLoader



class MergePartials:
    """Coordinates the multiprocessing """
    def __init__(self, path):

        n_bm  =  np.load(path + '/lbl_nuclei_bm.npz')['arr_0'].astype(np.uint16)                 # load the various matrices written during the analysis
        n_dm  =  np.load(path + '/lbl_nuclei_dm.npz')['arr_0'].astype(np.uint16)
        n_am  =  np.load(path + '/lbl_nuclei_am.npz')['arr_0'].astype(np.uint16)
        n_sg  =  np.load(path + '/lbl_segm_dm.npz')['arr_0'].astype(np.uint16)

        conc_wild                      =  np.zeros((n_bm.shape[0] + n_dm.shape[0] + n_am.shape[0] - 2, n_bm.shape[1], n_bm.shape[2]), dtype=np.uint16)     # initialization of the final matrix
        conc_wild[:n_bm.shape[0] - 1]  =  n_bm[:-1]                                                                                 # the starting labels are the one of the before mitosis matrix

        # the last frame of during mitosis is the same as the first of during mitosis, as well as the last frame of during mitosis is the
        # same frame as the first of after mitosis. This helps a lot in updating the label in order to make each nucleus keep the same
        # label during the whole evolution.

        # n_dm2    =  np.zeros(n_dm.shape, dtype=np.uint16)                                                                # initialization of the matrix of during mitosis with labels matching with the before mitosis
        idxs_dm  =  np.unique(n_dm[0, :, :])[1:]

        cpu_ow      =  multiprocessing.cpu_count()
        idxs_chops  =  np.array_split(idxs_dm, cpu_ow)

        job_args  =  []
        for k in range(cpu_ow):
            job_args.append([n_dm, idxs_chops[k], n_am, n_sg])        # in the multiprocessing pool each core will work on certain indexes: here we chop the frames

        pool     =  multiprocessing.Pool()
        results  =  pool.map(MergePartialUtility, job_args)
        pool.close()

        n_dm2  =  results[0].n_dm2
        for k in range(1, len(results)):
            if results[k].n_dm2.shape[0] != 0:
                n_dm2  +=  results[k].n_dm2 + np.sign(results[k].n_dm2) * (n_dm2.max() + 1)

        n_dm2  =  label(n_dm2, connectivity=1)

        results  =  0
        del results

        conc_wild[n_bm.shape[0] - 1:n_bm.shape[0] - 1 + n_dm.shape[0], :, :]  =  n_dm2                  # concatenate before and during mitosis with not matching labels (useful in other parts of the code)
        conc_wild[n_bm.shape[0] - 1 + n_dm.shape[0]:, :, :]                   =  n_am[1:, :, :]         # conc wild are nuclei keeping tracking in the 3 parts separately, but not between them: systers have different tags

        conc_nuc                            =  np.zeros(conc_wild.shape, dtype=np.int16)
        conc_nuc[:n_bm.shape[0] - 1, :, :]  =  n_bm[:-1, :, :]

        idxs1  =  np.unique(n_bm[-1, :, :])[1:]
        pbar   =  ProgressBar(total1=idxs1.size)
        pbar.show()

        for k in idxs1:
            pbar.update_progressbar(k)
            msk  =  (n_bm[-1, :, :] == k) * n_dm[0, :, :]
            id2  =  msk.max()
            if id2 != 0:
                conc_nuc[n_bm.shape[0] - 1:n_bm.shape[0] + n_dm.shape[0] - 1, :, :]  +=  k * (n_dm == id2)

        t_mtss  =  n_bm.shape[0] + n_dm.shape[0] - 1
        idxs2   =  np.unique(conc_nuc[t_mtss - 1, :, :])[1:]

        pbar.update_progressbar(0)

        for k in idxs2:
            # pbar.update_progressbar(k)
            msk  =  (conc_nuc[t_mtss - 1, :, :] == k) * n_am[0, :, :]
            msk  =  remove_small_objects(msk, 50)
            id3  =  np.unique(msk)[1:]

            if id3.size > 0:
                for j in range(id3.size):
                    conc_nuc[t_mtss:, :, :]  +=  k * (n_am[1:, :, :] == id3[j])

        s_bm  =  SpotsFeaturesLoader.SpotsFeaturesLoader(path + '/trk_spots_bm_coords.npy').spts_mtx                      # merging all the spots without labels
        s_dm  =  SpotsFeaturesLoader.SpotsFeaturesLoader(path + '/trk_spots_dm_coords.npy').spts_mtx
        s_am  =  SpotsFeaturesLoader.SpotsFeaturesLoader(path + '/trk_spots_am_coords.npy').spts_mtx

        conc_spt  =  np.concatenate((np.sign(s_bm[:-1, :, :]), np.sign(s_dm), np.sign(s_am[1:, :, :])), axis=0)     # final concatenation of the spots of the three parts
        pbar.close()


        self.conc_wild  =  conc_wild
        self.conc_nuc   =  conc_nuc
        self.conc_spt   =  conc_spt
        self.frames_bm  =  s_bm.shape[0]
        self.frames_dm  =  s_dm.shape[0]



class MergePartialUtility:
    """Utility for the multiprocessing."""
    def __init__(self, arg):

        n_dm     =  arg[0]
        idxs_dm  =  arg[1]
        n_am     =  arg[2]
        n_sg     =  arg[3]
        n_dm2    =  np.zeros(n_dm.shape, dtype=np.uint16)

        for k in range(idxs_dm.size):
            aa  =  DuringMitosisDaughterLabel.DuringMitosisDaughterLabel(n_dm, k, idxs_dm, n_sg)               # define a sublabel for the daughters
            if aa.im_lbls_fin.max() == 2:
                o                    =   np.where(aa.im_lbls_fin.max(2).max(1) == 1)[0][-1]
                n_dm2[:o + 1, :, :]  +=  idxs_dm[k] * (n_dm[:o + 1, :, :] == idxs_dm[k])

                ref1                 =   (aa.im_lbls_fin[-1, :, :] == 1) * n_am[0, :, :]                 # check the label that daughter one has in after mitosis
                ref1                 =   np.sort(ref1.reshape(ref1.size))
                ref1                 =   ref1[np.where(ref1 == 0)[0][-1] + 1:]
                ref1                 =   np.nan_to_num(np.median(ref1)).astype(np.uint16)
                n_dm2[o + 1:, :, :]  +=  ref1 * (aa.im_lbls_fin[o + 1:, :, :] == 1)                     # add the daughter with the correct label

                ref2                 =   (aa.im_lbls_fin[-1, :, :] == 2) * n_am[0, :, :]                # check the label that daughter two has in after mitosis
                ref2                 =   np.sort(ref2.reshape(ref2.size))
                ref2                 =   ref2[np.where(ref2 == 0)[0][-1] + 1:]
                ref2                 =   np.nan_to_num(np.median(ref2)).astype(np.uint16)
                n_dm2[o + 1:, :, :]  +=  ref2 * (aa.im_lbls_fin[o + 1:, :, :] == 2)                     # add the daughter with the correct label

            else:
                n_dm2  +=  (aa.im_lbls_fin[-1, :, :] * n_am[0, :, :]).max() * aa.im_lbls_fin

            self.n_dm2  =  n_dm2



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




