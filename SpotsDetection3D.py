"""This function detects spots in the 4D (time-x-y-z) stack.

This function is supposed to work in a multiprocessing pool. in_args are the
raw data (4D matrix) plus a threshold value. They are organized in a
list for multiprocessing purpose.
"""

import numpy as np
from scipy.ndimage import filters
from scipy.stats import norm
from skimage.morphology import label
from skimage.measure import regionprops

import SpotsDetectionUtility


class SpotsDetection3D:
    """Class working on several time frames"""
    def __init__(self, in_args):                                                 # for multiprocessing purposes I need to define a single in_args variable which is a list. The relative class will act consequently

        green4D         =  in_args[0]
        thr_val         =  in_args[1]
        volume_thr_var  =  in_args[2]

        steps, zlen, xlen, ylen  =  green4D.shape

        spots_ints    =  np.zeros((steps, xlen, ylen), dtype=np.uint16)
        spots_vol     =  np.zeros((steps, xlen, ylen), dtype=np.int16)
#         spots_lbl     =  np.zeros((steps, zlen, xlen, ylen), dtype=np.int16)
        spots_coords  =  np.zeros((0, 4), dtype=np.int16)
        spots_tzxy    =  np.zeros((0, 4), dtype=np.int16)

        for t in range(steps):
            g21          =  green4D[t, :, :, :]                                           # for each time step, we Gaussian filter the 3D stack (x-y-z) and than Laplacian filter
            g21g         =  filters.gaussian_filter(g21.astype(np.float), 1)
            g21f         =  filters.laplace(g21g.astype(np.float))
            (mu, sigma)  =  norm.fit(np.abs(g21f))                                        # histogram is fitted with a Gaussian function
            g21f_thr     =  np.abs(g21f) > mu + thr_val * sigma                           # thresholding on the histogram
            g21f3dlbl    =  label(g21f_thr).astype(np.int16)                              # labelling

            i_in       =  []                                                              # tags of spots that satisfies the conditions (volume and z planes)
            i_out      =  []                                                              # tags of spots that don't satisfies the conditions
            rgp_discr  =  regionprops(g21f3dlbl)
            for k in range(len(rgp_discr)):
                if rgp_discr[k]['area'] > volume_thr_var and np.diff(np.sort(rgp_discr[k]['coords'][:, 0])).sum() > 0:     # in 3D 'area' is volume; with 'coords' we plot all the values of the z coordinates of all the pixels. 
                    i_in.append(rgp_discr[k]['label'])                                                                     # than we and and calculate the sum of the derivative: if it is zero, and one z and and so on
                    spots_tzxy  =  np.concatenate([spots_tzxy, [np.append(np.int16(t), np.round(np.asarray(rgp_discr[k]['centroid'])).astype(np.int16))]], axis=0)    # coordinates of the good spots
                else:
                    i_out.append(rgp_discr[k]['label'])

            zz              =  SpotsDetectionUtility.spts_int_vol(g21f3dlbl.astype(np.int), g21.astype(np.int), i_in)             # output are 3 matrices: spot intensity summed in z, volume summed in z and 3D spots to remove
            spots_ints[t]  +=  zz[1].astype(np.uint16)
            spots_vol[t]   +=  zz[0].astype(np.int16)
#             spots_lbls[t]  +=  zz[2].astype(np.int16)

            rgp  =  regionprops(zz[2])
#            coords_bff  =  np.zeros((0, 3), np.int16)
            for rr in rgp:
                spots_coords  =  np.concatenate((spots_coords, np.column_stack((t * np.ones(rr['coords'].shape[0]), rr['coords'])).astype(np.int16)), axis=0)
#                 coords_bff  =   np.concatenate((coords_bff, rr['coords']), axis=0)

        self.spots_ints  =  spots_ints
        self.spots_vol   =  spots_vol
        self.spots_tzxy  =  spots_tzxy
#         self.spots_lbls  =  spots_lbls
        self.spots_coords  =  spots_coords


class SpotsDetection3D_Single:
    """Class working on a single time frame"""
    def __init__(self, in_args):                                                      # for multiprocessing purposes I need to define a single in_args variable which is a list. The relative class will act consequently

        green4D         =  in_args[0]
        thr_val         =  in_args[1]
        volume_thr_var  =  in_args[2]

        xlen, ylen   =  green4D.shape[1:]
        spots_ints   =  np.zeros((xlen, ylen), dtype=np.int32)
        spots_vol    =  np.zeros((xlen, ylen), dtype=np.int16)
        g21          =  green4D                                                       # for each time step, we Gaussian filter the 3D stack (x-y-z) and than Laplacian filter
        g21g         =  filters.gaussian_filter(g21, 1)
        g21f         =  filters.laplace(g21g.astype(np.float))
        (mu, sigma)  =  norm.fit(np.abs(g21f))                                        # histogram is fitted with a Gaussian function
        g21f_thr     =  np.abs(g21f) > mu + thr_val * sigma                           # thresholding on the histogram
        g21f3dlbl    =  label(g21f_thr)                                               # labelling
        g2show       =  np.zeros(g21f3dlbl.shape)

        i_in       =  []                                                              # tags of spots that satisfies the conditions (volume and z planes)
        i_out      =  []                                                              # tags of spots that don't satisfies the conditions
        rgp_discr  =  regionprops(g21f3dlbl)
        for k in range(len(rgp_discr)):
            if rgp_discr[k]['area'] > volume_thr_var and np.diff(np.sort(rgp_discr[k]['coords'][:, 0])).sum() > 0:     # in 3D 'area' is volume; with 'coords' we plot all the values of the z coordinates of all the pixels. 
                i_in.append(rgp_discr[k]['label'])                                                                     # than we and and calculate the sum of the derivative: if it is zero, and one z and and so on
                g2show  +=  g21f3dlbl == rgp_discr[k]['label']
            else:
                i_out.append(rgp_discr[k]['label'])

        zz           =  SpotsDetectionUtility.spts_int_vol(g21f3dlbl.astype(np.int), g21.astype(np.int), i_in, i_out)             # output are 3 matrices: spot intensity summed in z, volume summed in z and 3D spots to remove
        spots_ints  +=  zz[1]
        spots_vol   +=  zz[0]

        self.spots_ints  =  spots_ints
        self.spots_vol   =  spots_vol
        self.spots_lbls  =  np.sign(g2show)


