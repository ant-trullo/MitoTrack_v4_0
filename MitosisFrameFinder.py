"""Given the time series of a single nucleus (extracted from the whole stack) evolving from the beginning 13th cycle up to the end of the 14th,
   this function finds the time frame of the mitosis and the coordinate of the center of mass of the nucleus during mitosis."""

import numpy as np
from scipy import ndimage
from skimage.morphology import label, remove_small_objects    # , remove_small_holes
from skimage.measure import regionprops


class MitosisFrameFinder:
    """The only class, does all the job"""
    def __init__(self, nuc):

        prof  =  np.zeros(nuc.shape[0])

        for t in range(prof.size):
            bff      =  nuc[t]
            prof[t]  =  np.unique(bff[bff != 0]).size

        if prof.max() == 2:
            j         =  np.where(prof == 2)[0][0]
            mpts1     =  j                                                          # mtps1 is the first frame in which the daugheters are split
            mpts2     =  j                                                          # mtps2 is the mitosis frame
            num_lbls  =  1
            iters     =  1

            """Sometime deformation during mitosis can mislead the software: to avoid this problem, the algorithm checks that with binary
            erosion the single connected component does not become 1. the number of iteration of the erosion is kept to 7 simply because
            this number works on the data seen up to now"""

            while iters < 7 and num_lbls < 1.5:
                img       =   ndimage.binary_erosion(nuc[j - 1, :, :], iterations=iters)     # binary erosion
                img_lbls  =   label(img)
                img_lbls  =   remove_small_objects(img_lbls, 10)                            # small object are removed to avoid misleading of the software    
                img_lbls  =   label(img_lbls)                                               # labeling of the eroded connected component
                num_lbls  =   img_lbls.max()                                                # number of connected components
                iters     +=  1

            rgp  =  regionprops(img_lbls)
            if len(rgp) > 1:
                a_vec  =  np.zeros((len(rgp)))
                for k in range(a_vec.size):
                    a_vec[k]  =  rgp[k]['area']
                a_vec.sort()
                a1, a2  =  a_vec[-1], a_vec[-2]

                if a1 / a2 < 4:
                    mpts2  =  j - 1

        else:
            mpts2  =  0

        self.mpts  =  [mpts1, mpts2]






# """Given the time series of a single nucleus (extracted from the whole stack) evolving from the beginning 13th cycle up to the end of the 14th,
#    this function finds the time frame of the mitosis and the coordinate of the center of mass of the nucleus during mitosis."""
# 
# import numpy as np
# from scipy import ndimage
# from skimage.morphology import label, remove_small_objects, remove_small_holes
# from skimage.measure import regionprops
# 
# 
# class MitosisFrameFinder:
#     """The only class, does all the job"""
#     def __init__(self, nuc):
# 
#         prof  =  np.zeros(nuc.shape[0])
# 
#         for t in range(prof.size):
#             prof[t]  =  label(nuc[t], connectivity=1).max()       # number of connected components frame by frame
#             nuc[t]   =  remove_small_holes(nuc[t], 40)            # cleaning the nuc matrix
# 
#         mpts  =  0
# #         ctrs_mts  =  np.zeros(2)
#         if prof[0] == 2:                                                    # removes errors (the nucleus cannot start with a mitosis)
#             prof[0] = 1
# 
#         if prof.max() > 1:                                                  # check that it become two or not (sometimes we have anomalous nuclei)
#             j         =  np.where(prof == 2)[0][0]                          # time frame in which mother becomes two daughters
#             mpts      =  j
#             num_lbls  =  1
#             iters     =  1
# 
# 
#         # sometime deformation during mitosis can mislead the software: to avoid this problem, the algorithm checks that with binary
#         # erosion the single connected component does not become 1. the number of iteration of the erosion is kept to 7 simply because
#         # this number works on the data seen up to now 
# 
# 
#             while iters < 7 and num_lbls < 1.5:
#                 img       =   ndimage.binary_erosion(nuc[j - 1, :, :], iterations=iters)     # binary erosion
#                 img_lbls  =   label(img)
#                 img_lbls  =   remove_small_objects(img_lbls, 10)                            # small object are removed to avoid misleading of the software    
#                 img_lbls  =   label(img_lbls)                                               # labeling of the eroded connected component
#                 num_lbls  =   img_lbls.max()                                                # number of connected components
#                 iters     +=  1
# 
#             rgp  =  regionprops(img_lbls)
#             if len(rgp) > 1:
#                 a_vec  =  np.zeros((len(rgp)))
#                 for k in range(a_vec.size):
#                     a_vec[k]  =  rgp[k]['area']
#                 a_vec.sort()
#                 a1, a2  =  a_vec[-1], a_vec[-2]
# 
#                 if a1 / a2 < 4:
#                     mpts  =  j - 1
# 
# #             rgp_fin   =  regionprops(nuc[mpts, :, :])                   # measure the centroid of the nucleus just before mitosis
# #             ctrs_mts  =  rgp_fin[0]['centroid']
# # 
#         self.mpts      =  mpts
# #         self.ctrs_mts  =  ctrs_mts
# 
# 
