"""This function loads the partial analysis.

Because for the green channel we save the 4D matrix,
when loaded this channel is maximum intensity projected
for visualization in the main GUI.
"""


import numpy as np

import SpotsFeaturesLoader


class LoadPartialAnalysisFrstChop:
    """Load partial analysis of before mitosis"""
    def __init__(self, folder_storedata):

        imarray_red      =  np.load(folder_storedata + '/imarray_red_whole.npy')
        green4D          =  np.load(folder_storedata + '/green4D_whole.npy')
        mtss_boundaries  =  np.load(folder_storedata + '/mtss_boundaries.npy')

        imarray_red    =  imarray_red[:mtss_boundaries[0] + 1]
        green4D        =  green4D[:mtss_boundaries[0] + 1]

        t_steps, zlen, xlen, ylen   =  green4D.shape
        imarray_green  =  np.zeros((t_steps, xlen, ylen))
        for t in range(t_steps):
            for x in range(xlen):
                imarray_green[t, x, :]  =  green4D[t, :, x, :].max(0)

        self.imarray_red    =  imarray_red
        self.imarray_green  =  imarray_green
        self.green4D        =  green4D



class LoadPartialAnalysisScndChop:
    """Load partial analysis of during mitosis"""
    def __init__(self, folder_storedata):

        imarray_red      =  np.load(folder_storedata + '/imarray_red_whole.npy')
        green4D          =  np.load(folder_storedata + '/green4D_whole.npy')
        mtss_boundaries  =  np.load(folder_storedata + '/mtss_boundaries.npy')

        imarray_red    =  imarray_red[mtss_boundaries[0]:mtss_boundaries[1]]
        green4D        =  green4D[mtss_boundaries[0]:mtss_boundaries[1]]

        t_steps, zlen, xlen, ylen   =  green4D.shape
        imarray_green  =  np.zeros((t_steps, xlen, ylen))
        for t in range(t_steps):
            for x in range(xlen):
                imarray_green[t, x, :]  =  green4D[t, :, x, :].max(0)


        self.imarray_red    =  imarray_red
        self.imarray_green  =  imarray_green
        self.green4D        =  green4D



class LoadPartialAnalysisThrdChop:
    """Load partial analysis of after mitosis"""
    def __init__(self, folder_storedata):

        imarray_red      =  np.load(folder_storedata + '/imarray_red_whole.npy')
        green4D          =  np.load(folder_storedata + '/green4D_whole.npy')
        mtss_boundaries  =  np.load(folder_storedata + '/mtss_boundaries.npy')

        imarray_red    =  imarray_red[mtss_boundaries[1] - 1:]
        green4D        = green4D[mtss_boundaries[1] - 1:]

        t_steps, zlen, xlen, ylen   =  green4D.shape
        imarray_green  =  np.zeros((t_steps, xlen, ylen))
        for t in range(t_steps):
            for x in range(xlen):
                imarray_green[t, x, :]  =  green4D[t, :, x, :].max(0)

        self.imarray_green  =  imarray_green
        self.imarray_red    =  imarray_red
        self.green4D        =  green4D



class LoadSpots:
    """Load partial analysis of after mitosis"""
    def __init__(self, folder_storedata, flag):

#         spots_ints  =  np.load(folder_storedata + '/ints_spots_' + flag + 'm.npy')
#         spots_vol   =  np.load(folder_storedata + '/vol_spots_' + flag + 'm.npy')
        self.spots_tzxy  =  np.load(folder_storedata + '/tzxy_spots_' + flag + 'm.npy')
        self.spots_ints  =  SpotsFeaturesLoader.SpotsFeaturesLoader(folder_storedata + '/ints_spots_' + flag + 'm_coords.npy').spts_mtx
        self.spots_vol   =  SpotsFeaturesLoader.SpotsFeaturesLoader(folder_storedata + '/vol_spots_' + flag + 'm_coords.npy').spts_mtx

#         self.spots_ints  =  spots_ints
#         self.spots_vol   =  spots_vol
#         self.spots_tzxy  =  spots_tzxy







