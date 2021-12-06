
"""This function trackes nuclei across mitosis.

Given a time series of segmented nuclei, this function trackes nuclei using
overlapping arguments up to the moment in which from a nucleus there are tools
showing the same overlapping (daughters). Once this point is found, the algorithm
breaks and the rest of tracking is done nucleus by nucleus with a normal tracking
based on minimum distance criterion. All the tracked nuclei are checked in terms
of number of connected components (1 at the beginning and two at the end).
"""


import numpy as np
from skimage.morphology import label, remove_small_objects, binary_dilation
from PyQt5 import QtCore, QtGui

import NucleiConnect4MTool


class MitosisCrossing3:
    def __init__(self, lbl_start, overlap_ratio, prolif_ratio, size_thr, nuc_dist):

        lbl_start  =  lbl_start.astype(np.int)                                                                  # just makes the tag integers if they are not
        steps      =  lbl_start.shape[0]                                                                        # number of time steps
        lbl_tot    =  label(np.sign(lbl_start))                                                                 # 3D labelling, results to be discriminate

        tot_lbls  =  np.zeros(lbl_tot.shape)
        for t in range(steps):                                                                                  # 2D labelling, frame by frame, to isolate all the speckles
            tot_lbls[t, :, :]  =  label(lbl_start[t, :, :], connectivity=1)
            tot_lbls[t, :, :]  =  remove_small_objects(lbl_start[t, :, :], size_thr)                                  # remove small connected components coming from segmenttation

        lidx  =  np.unique(lbl_tot)[1:]                                                                         # array with all the nuclei tags

        pbar  =  QtGui.QProgressBar()
        pbar.setRange(0, 0)
        pbar.show()

        mt_trk_frst  =  np.zeros(lbl_tot.shape)                                                                # initialization of the first part of the final matrix
        l            =  1
        for k in range(lidx.size):
            nn    =  np.zeros((steps))                                                                          # discrimination: for each label we find in the 3D labelling, we verify that it has one connected object before mitosis
            QtCore.QCoreApplication.processEvents()
            maskt  =  (lbl_tot == lidx[k]).astype(np.int)                                                       # labeling of the single mother and its daughters
            for t in range(steps):
                nn[t]  =  label(maskt[t, :, :], connectivity=1).max()

                if nn[0] == 1 and nn[-1] == 2 and np.sum(np.abs(np.diff(nn[:]))) == 1:                          # if the number of connected components is 1 at the biginning, 2 at the end and does not change anymore, daughters are correctly associated to the mother
                    mt_trk_frst  +=  l * (lbl_tot == lidx[k]).astype(np.int)
                    l            +=  1


        tot_lbls     *=  (1 - np.sign(mt_trk_frst)).astype(np.int)                                      # matrix of the untracked nuclei
        idxs_left    =   np.unique(tot_lbls[0, :, :])[1:]                                               # array with the tags of the nuclei still to track of the first time frame
        new_lab      =   mt_trk_frst.max() + 1                                                          # first free tag to associate to the nuclei to be tracked
        mt_trk_scnd  =   np.zeros(mt_trk_frst.shape)                                                    # initialization of the matrix of the nuclei to be tracked

        for bb in idxs_left:
            QtCore.QCoreApplication.processEvents()
            bfr_msks           =  np.zeros(tot_lbls.shape)                                          # initialization of the matrix for a single mother daughters
            msk                =  (tot_lbls[0, :, :] == bb).astype(np.int)                          # B&W calc of the mother in the first frame
            bfr_msks[0, :, :]  =  msk

            for t in range(1, steps):
                mref  =  (msk * tot_lbls[t, :, :])                                                  # the calc is multiplied with the following frame to check which nucleus (or nuclei) are overlapped
                if mref.sum() == 0 and msk.sum() > 0:                                               # if there is no overlap, the calc is dilated to better check the overlap
                    mref  =  (binary_dilation(msk, selem=np.ones((5, 5))) * tot_lbls[t, :, :])

                mref  =  np.trim_zeros(np.sort(mref.flatten()))                                     # from the multiplication we extract the values of the matrix of the overlap (tags)
                if np.unique(mref).size == 1 and mref.size > 50:                                    # we check the overlap is present and is consistent (more than 50 pixels surface)
                    fl                 =   mref[0]                                                  # if there is just one number (tag) in the overlapping, means that mitosis is not there still
                    bfr_msks[t, :, :]  +=  (tot_lbls[t, :, :] == fl)                                # only one connected component is associated
                    msk                =   (tot_lbls[t, :, :] == fl)

                elif np.unique(mref).size == 2:                                                                                                                      # if instead there are two tags, we check the overlapped surfaces are not too different for both tags

                    mref_lbls  =  label(mref, connectivity=1)                                                                                                        # the overlap ratio is the numerical control for this
                    if 1.0 * np.min([np.sum(mref_lbls == 1), np.sum(mref_lbls == 2)]) / np.max([np.sum(mref_lbls == 1), np.sum(mref_lbls == 2)]) < overlap_ratio:    # if the overlap ratio is smaller than the defined one, the smaller overlapping happend by chance (other nuclei are close)
                        fl                 =   np.round(np.median(mref))                                                                                             # and the overlapping is considered for only the biggest surface
                        bfr_msks[t, :, :]  +=  (tot_lbls[t, :, :] == fl)
                        msk                =   (tot_lbls[t, :, :] == fl)

                    if 1.0 * np.min([np.sum(mref_lbls == 1), np.sum(mref_lbls == 2)]) / np.max([np.sum(mref_lbls == 1), np.sum(mref_lbls == 2)]) > prolif_ratio:     # if instead the two sufaces are similar in size, we consider the mitosis is happening and so the labelis associatedc to the two connected components
                        mref               =   np.unique(mref).astype(np.int)
                        fl                 =   mref[0]
                        bfr_msks[t, :, :]  +=  (tot_lbls[t, :, :] == fl)
                        msk                =   (tot_lbls[t, :, :] == fl)
                        fl                 =   mref[1]
                        bfr_msks[t, :, :]  +=  (tot_lbls[t, :, :] == fl)
                        msk                +=  (tot_lbls[t, :, :] == fl)

                        break

                elif np.unique(mref).size > 2 or np.unique(mref).size == 0:                                                                                     # if the number of connected components overlapping is 3, something is going wrong and there is no association. It is left for manual corrections
                    break

            mt_trk_scnd  +=  new_lab * bfr_msks                                         # adding new labels
            new_lab      +=  1


        # the previous part goes from frame zero up to the mitosis: in order to finish the work, we use the NucleiConnect4MTool
        # that will act from the frame in which the tracking stoped up to the end of the matrix


        lidx2        =  np.unique(mt_trk_scnd)[1:].astype(np.int)                       # array with the indexes of the tracked nuclei
        mt_trk_thrd  =  np.zeros(mt_trk_scnd.shape)                                     # initialization of the matrix with the last nuclei to track

        for k in range(lidx2.size):
            QtCore.QCoreApplication.processEvents()
            bfr_msks2  =   np.zeros(tot_lbls.shape)
            bfr_prf    =   (mt_trk_scnd == lidx2[k]).sum(2).sum(1)                                                                          # singl nucleus matrix
            t          =   np.where(bfr_prf == 0)[0]                                                                                        # time frame in which the tracking stops

            if t.size > 0 and t[0] < steps and t[0] > 0:                                                                                    # check that the time detected is not at the end or at the beginning
                t                    =   t[0]
                bfr_msks2[:t, :, :]  =   (mt_trk_scnd[:t, :, :] == lidx2[k])
                mref2                =   np.unique(bfr_msks2[t - 1, :, :] * tot_lbls[t - 1, :, :])[1:]                                      # tag of the nucleus in the original (input) matrix
                d1                   =   NucleiConnect4MTool.NucleiConnect4MTool(tot_lbls[t - 1:, :, :], mref2[0], nuc_dist).nucleus              # tracking via other class function for daughter 1
                bfr_msks2[t:, :, :]  +=  d1[1:, :, :]
                if mref2.size > 1:
                    d2                   =   NucleiConnect4MTool.NucleiConnect4MTool(tot_lbls[t - 1:, :, :], mref2[1], nuc_dist).nucleus          # tracking via other class function for daughter 2
                    bfr_msks2[t:, :, :]  +=  d2[1:, :, :]

                mt_trk_thrd  +=  lidx2[k] * bfr_msks2


        lidx3  =  np.unique(mt_trk_thrd)[1:]
        for k in range(lidx3.size):
            QtCore.QCoreApplication.processEvents()
            nn3    =  np.zeros((steps))                                                                         # discrimination: for each label we verify that it has one connected object before mitosis, two after and that this number does not change anymore
            maskt  =  (mt_trk_thrd == lidx3[k]).astype(np.int)
            for t in range(steps):
                nn3[t]  =  label(maskt[t, :, :], connectivity=1).max()

            if nn3.max() == 2:
                t_two     =  np.where(nn3 == 2)[0][0]
                nn3_diff  =  np.diff(nn3[t_two:])
                if nn3_diff.sum() <= 0:
                    mt_trk_frst  +=  lidx3[k] * (mt_trk_thrd == lidx3[k]).astype(np.int)


        mt_trk_scnd *=  (1 - np.sign(mt_trk_frst))
        mtx_tot     =   mt_trk_frst + mt_trk_scnd                                                     # nuclei still not tracked are left for manual tracking

        pbar.close()

        self.mtx_tot      =  mtx_tot.astype(np.uint16)
        self.left4manual  =  tot_lbls * (1 - np.sign(mtx_tot)).astype(np.uint16)


