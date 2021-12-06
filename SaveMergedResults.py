"""This function writes results of MergePartial in .bin files.

It takes as input the folder path in which save the data.
 """



import numpy as np
import SpotsFeaturesSaver


class SaveMergedResults:
    """The only class, does all the job"""
    def __init__(self, folder_path, mm, nuc_active, evol_check, spots_tracked):

        SpotsFeaturesSaver.SpotsFeaturesSaver(mm.conc_spt, folder_path + '/conc_spt_coords.npy')
        SpotsFeaturesSaver.SpotsFeaturesSaver(spots_tracked, folder_path + '/spots_tracked_coords.npy')

        np.savez_compressed(folder_path + '/merged_mm.npz', a=mm.conc_nuc, b=mm.conc_wild)
        np.savez_compressed(folder_path + '/merged_nuc_active.npz', a=nuc_active.nuclei_active, b=nuc_active.nuclei_active3c)
        np.savez_compressed(folder_path + '/merged_evol_check.npz', a=evol_check.conc_nuc_ok, b=evol_check.conc_nuc_wrong)


