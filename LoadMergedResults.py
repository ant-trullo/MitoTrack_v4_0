"""This function loads analysis saved after merge partial results.

It takes as input the path of the folder with the analysis and
reads the .npy files previously written.
 """


import numpy as np
from skimage.morphology import label

import SpotsFeaturesLoader



class LoadMergedResults_mm:
    """Load concatenate nuclei and spots"""
    def __init__(self, folder_path):

        self.conc_spt   =  SpotsFeaturesLoader.SpotsFeaturesLoader(folder_path + '/conc_spt_coords.npy').spts_mtx

        conc            =  np.load(folder_path + '/merged_mm.npz')
        self.conc_nuc   =  conc['a']
        self.conc_wild  =  conc['b']



class LoadMergedResultsNucActive:
    """Load active nuclei and generate activation curve"""
    def __init__(self, folder_path):

        nuc_act               =  np.load(folder_path + '/merged_nuc_active.npz')
        self.nuclei_active    =  nuc_act['a']
        self.nuclei_active3c  =  nuc_act['b']


        steps  =  self.nuclei_active.shape[0]

        self.n_active_vector  =  np.zeros((steps))
        for t in range(steps):
            self. n_active_vector[t]  =  label(self.nuclei_active[t, :, :] == 2).max()



class LoadMergedResultsEvolCheck:
    """Load result of evolution controller"""
    def __init__(self, folder_path):

        evol_conc            =  np.load(folder_path + '/merged_evol_check.npz')
        self.conc_nuc_ok     =  evol_conc['a']
        self.conc_nuc_wrong  =  evol_conc['b']



