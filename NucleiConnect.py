"""This function connects (tracks) nuclei.


It coordinates the action of the
'NucleiConnectSingle' function which act on the single nucleus in order to let it work
on the whole nuclei of the matrix-image. Input arguments are the matrix-image of the
nuclei and the distance threhsold. Nuclei in consecutive frames are connected following
the minimal distance criterion.
"""



import numpy as np
from skimage.measure import regionprops


import NucleiConnectSingle


class NucleiConnect:
    def __init__(self, input_args):

        nuclei    =  input_args[0].astype(np.int)
        dist_thr  =  input_args[1]

        t_tot  =  nuclei.shape[0]                               # total number of time frames
        ctrs   =  np.zeros((t_tot, 2, nuclei.max()))            # initialization of the matrix with the centroids of the nuclei

        for tt in range(t_tot):
            rgp    =  regionprops(nuclei[tt, :, :].astype(np.int))                                                                              # regionprops of the nuclei
            for j in range(len(rgp)):
                if nuclei[tt, np.round(rgp[j]['Centroid'][0]).astype(np.int), np.round(rgp[j]['Centroid'][1]).astype(np.int)] > 0:              # sometimes, because of shape irregularities or detection problems, the centroid is not inside the nucleus.
                    ctrs[tt, :, j]  =  rgp[j]['Centroid']                                                                                       # check if the centroid is in the nucleus
                else:
                    bff_square  =  nuclei[tt, np.round(rgp[j]['Centroid'][0]).astype(np.int) - 1:np.round(rgp[j]['Centroid'][0]).astype(np.int) + 2, np.round(rgp[j]['Centroid'][1]).astype(np.int) - 1:np.round(rgp[j]['Centroid'][1]).astype(np.int) + 2]     # if the centroid is not in the nucleus
                    bff_square  =  np.unique(bff_square)                                                                                                                                                                                                        # a square is drawn around the centroid searching for the closer non-zero value

                    bff_square  =  np.trim_zeros(bff_square)
                    if bff_square.size > 0:
                        bff_square  =  bff_square[0]
                    if bff_square > 0:
                        nuclei[tt, np.round(rgp[j]['Centroid'][0]).astype(np.int), np.round(rgp[j]['Centroid'][1]).astype(np.int)]  =  bff_square
                        ctrs[tt, :, j]  =  rgp[j]['Centroid']


        ctrs_data       =  np.copy(ctrs)
        nuclei_tracked  =  np.zeros(nuclei.shape, dtype=np.int16)                               # initialization of the matrix with tracked nuclei

        k  =  0
        while ctrs[:-1, :, :].sum() > 0:
            [t, i_ref]  =  np.argwhere(ctrs.sum(1) != 0)[0]
            bffr_data   =  NucleiConnectSingle.NucleiConnectSingle(ctrs, nuclei, i_ref, t, dist_thr)      # algorithm to track a single nucleus

            if bffr_data.labbs2.sum() > 0:                                  # check the tracking worked on the nucleus
                nuclei_tracked  +=  (k + 1) * bffr_data.labbs2              # if it worked, the single tracked nucleus is added to the final time series with the proper tag
                ctrs            =   bffr_data.ctrs
                k               +=  1


        self.nuclei_tracked         =  nuclei_tracked
        self.ctrs                   =  ctrs
        self.ctrs_data              =  ctrs_data
