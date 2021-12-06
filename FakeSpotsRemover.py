"""This function deletes spots which appears just for an isolated time frame.

If a spots is present in one frame and is not in the previous and in the following,
the spot is removed. The input of this function is the time series of the tracked spots
in the before-after or during mitosis part, the output is the same time series but filtered.

FakeSpotsRemoverMerged does the same job but is optimized for the merged time series.

"""


import numpy as np
from skimage.morphology import label

import SpotsConnection


class FakeSpotsRemover:
    """Removes the spots activating for only one frame in the single cycle"""
    def __init__(self, spts_tracked):

        idxs        =  np.unique(spts_tracked)[1:]                                  # list of the spot tags in the time series
        delete_msk  =  np.zeros(spts_tracked.shape)                                 # initialization of the matrix with the deleted spots

        for k in idxs:
            profile  =  np.sign((spts_tracked == k).sum(2).sum(1))                      # presence trace (0 or 1 vector) of a spots
            prof_d2  =  np.diff(profile, 2)
            zero_jj  =  np.where(prof_d2 == -2)[0] + 1                                  # the second derivative gives -2 when the signal is ...,0, 1, 0, ... . It detects the isolated activation

            for l in zero_jj:                                                           # zero_jj is empty {In[1]: zero_jj; Out[1]: array([], dtype=int64)} the 'for' cycle do not even start
                delete_msk[l, :, :]  +=  (spts_tracked[l, :, :] == k)

        spts_tracked_filtered  =  spts_tracked * (1 - delete_msk)                       # removal of the spots

        self.spts_tracked_filtered  =  spts_tracked_filtered



class FakeSpotsRemoverMerged:
    """Removes the spots activating for only one frame in the merged stack"""
    def __init__(self, conc_spt, conc_nuc, conc_wild, frames_bm, frames_dm, max_dist_ns):

        spts_trck_bm  =  SpotsConnection.SpotsConnection(conc_nuc[:frames_bm], conc_spt[:frames_bm].astype(np.int), max_dist_ns).spots_tracked
        del_msk_bm    =  np.zeros(spts_trck_bm.shape, dtype=np.uint16)                                    # initialization of the matrix with the deleted spots
        idxs_bm       =  np.unique(spts_trck_bm)[1:]

        for k in idxs_bm:
            prof_lbls   =  label(np.sign(np.sum((spts_trck_bm == k), axis=(1, 2))))
            for kk in range(1, prof_lbls.max() + 1):
                if np.sum(prof_lbls == kk) == 1:
                    t_ref              =   np.where(prof_lbls == kk)[0][0]
                    del_msk_bm[t_ref]  +=  spts_trck_bm[t_ref] == k


        spts_trck_am  =  SpotsConnection.SpotsConnection(conc_wild[frames_bm + frames_dm - 1:], conc_spt[frames_bm + frames_dm - 1:].astype(np.int), max_dist_ns).spots_tracked
        del_msk_am    =  np.zeros(spts_trck_am.shape, dtype=np.uint16)                                    # initialization of the matrix with the deleted spots
        idxs_am       =  np.unique(spts_trck_am)[1:]

        for k in idxs_am:
            prof_lbls   =  label(np.sign(np.sum((spts_trck_am == k), axis=(1, 2))))
            for kk in range(1, prof_lbls.max() + 1):
                if np.sum(prof_lbls == kk) == 1:
                    t_ref              =   np.where(prof_lbls == kk)[0][0]
                    del_msk_am[t_ref]  +=  spts_trck_am[t_ref] == k

        del_msk_tot                              =   np.zeros(conc_nuc.shape, dtype=np.uint16)
        del_msk_tot[:frames_bm]                  +=  del_msk_bm
        del_msk_tot[frames_bm + frames_dm - 1:]  +=  del_msk_am

        conc_spt  *=  (1 - del_msk_tot)

        self.spots_tracked  =  SpotsConnection.SpotsConnection(conc_nuc, conc_spt.astype(np.int), max_dist_ns).spots_tracked




#class FakeSpotsRemoverMerged:
#    def __init__(self, spts_tracked):
#
#        idxs     =  np.unique(spts_tracked)[1:]                                     # list of the spot tags in the time series
#        del_msk  =  np.zeros(spts_tracked.shape)                                    # initialization of the matrix with the deleted spots
#
#        for k in idxs:
#            profile  =  (spts_tracked == k).sum(2).sum(1)
#            ones_jj  =  np.where(profile[:-1] == 1)[0]
#            for j in ones_jj:
#                if profile[j+1] <= 1:
#                    del_msk[j, :, :]  +=  (spts_tracked[j, :, :] == k)
#
#            twos_jj  =  np.where(profile[1:-2] == 2)[0]
#            for j in twos_jj:
#                if profile[j-1:j+2].sum() < 3:
#                    del_msk[j, :, :]  +=  (spts_tracked[j, :, :] == k)
#
#        spts_tracked_filtered       =  spts_tracked * (1 - del_msk)
#
#        self.spts_tracked_filtered  =  spts_tracked_filtered
