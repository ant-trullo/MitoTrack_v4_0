"""This function generates a video with different colors for mother active and mother inactive nuclei.

Input data is the folder path with analysis results.
"""


# from os import listdir
import numpy as np
# from xlrd import open_workbook
import tifffile
# from scipy.ndimage import binary_dilation

import LoadMergedResults


class FalseColoredMakeUp:
    """Only class, does all the job"""
    def __init__(self, folder_name):

        imarray_red_whole                            =  np.load(folder_name + '/imarray_red_whole.npy')
        imarray_red_whole[imarray_red_whole >= 700]  =  700
        nucs_act                                     =  np.load(folder_name + '/merged_nuc_active.npz')["a"] == 2
        nucs_inact                                   =  (np.sign(np.load(folder_name + "/merged_mm.npz")["a"]) * (1 - nucs_act)).astype(bool)
        mtx_fin                                      =  np.zeros(nucs_act.shape + (3,), np.uint16)
        con_spts                                     =  LoadMergedResults.LoadMergedResults_mm(folder_name).conc_spt

        mtx_fin[:, :, :, 0]  =  nucs_act  *  imarray_red_whole
        mtx_fin[:, :, :, 1]  =  nucs_act  *  imarray_red_whole
        mtx_fin[:, :, :, 2]  =  nucs_act  *  imarray_red_whole
        mtx_fin[:, :, :, 2]  +=  nucs_inact  *  imarray_red_whole

        mtx_fin[:, :, :, 0]  *=  (1 - con_spts)
        mtx_fin[:, :, :, 1]  *=  (1 - con_spts)
        mtx_fin[:, :, :, 2]  *=  (1 - con_spts)
        mtx_fin[:, :, :, 0]  +=  con_spts * imarray_red_whole.max()

        tifffile.imwrite(folder_name + '/active_inactive_makeup.tif', np.rot90(mtx_fin[:, :, ::-1], axes=(1, 2)))
        # tifffile.imwrite('/home/atrullo/Desktop/active_inactive_makeup.tif', np.rot90(mtx_fin[:, :, :-1], axes=(1, 2)))

        self.mtx_fin  =  mtx_fin


#         files  =  listdir(folder_name)                         # search in the analysis folder the xls files to read the background values
#         for file in files:
#             if file[:24] == "MemoryStudyWithDistances":        # select the .xls file with spot analysis results, green
#                 xls_name  =  file
#                 break
# 
#         imarray_red_whole  =  np.load(folder_name + '/imarray_red_whole.npy')
#         evol_conc_ok       =  LoadMergedResults.LoadMergedResultsEvolCheck(folder_name).conc_nuc_ok
# 
# 
#         book       =  open_workbook(folder_name + "/" + xls_name)
#         sheet      =  book.sheet_by_index(0)
#         tags_list  =  sheet.col_values(0)[1:]
#         mthr_list  =  sheet.col_values(1)[1:]
# 
#         kk  =  1
#         while tags_list[kk] != '':
#             kk += 1
# 
#         tags_list        =  tags_list[:kk]
#         mthr_list        =  mthr_list[:kk]
#         tags_list        =  [int(tag[3:])for tag in tags_list]
#         mthr_list        =  np.asarray(mthr_list)
#         mthr_act_tags    =  [tags_list[idx] for idx in np.where(mthr_list != 0)[0]]
#         mthr_inact_tags  =  [tags_list[idx] for idx in np.where(mthr_list == 0)[0]]
# 
#         mthr_act_inact_mtx  =  np.zeros(evol_conc_ok.shape + (3,))
#         msk                 =  np.zeros(imarray_red_whole.shape)
#         for k in mthr_act_tags:
#             bff                              =  (evol_conc_ok == k)
#             msk                             +=  bff
#             mthr_act_inact_mtx[:, :, :, 1]  +=  imarray_red_whole * bff
# 
#         for j in mthr_inact_tags:
#             bff                              =  (evol_conc_ok == j)
#             msk                             +=  bff
#             mthr_act_inact_mtx[:, :, :, 0]  +=  imarray_red_whole * bff
# 
#         for t in range(msk.shape[0]):
#             msk[t]  =  binary_dilation(msk[t], iterations=2)
# 
#         mthr_act_inact_mtx[:, :, :, 0]  *=  1 - con_spts     # + con_spts * imarray_red_whole.max()
#         mthr_act_inact_mtx[:, :, :, 1]  *=  1 - con_spts     # + con_spts * imarray_red_whole.max()
#         mthr_act_inact_mtx[:, :, :, 2]  *=  1 - con_spts     # + con_spts * imarray_red_whole.max()
#
#         mthr_act_inact_mtx[:, :, :, 0]  +=  con_spts * msk * imarray_red_whole.max()
#         mthr_act_inact_mtx[:, :, :, 1]  +=  con_spts * msk * imarray_red_whole.max()
#         mthr_act_inact_mtx[:, :, :, 2]  +=  con_spts * msk * imarray_red_whole.max()






