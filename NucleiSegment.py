"""This works on a single frame, with nuclei detected and labeled but not segmented.

Input are the frame with detected nuclei, the circularity estimated and the watershed parameter.
"""

import numpy as np
from skimage.morphology import label, remove_small_objects
from scipy import ndimage
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage import filters
from skimage.segmentation import watershed

import CircularityEstimate


class NucleiSegment:
    """The only class, does all the job"""
    def __init__(self, frame1, circ_thr, lmp):

        circ  =  CircularityEstimate.CircularityEstimate(frame1.astype(int)).circ        # circularity of all the nuclei estimate
        aa    =  np.where(circ[0, :] > circ_thr)[0]                                         # circularity thresholding

        frame1_sgm  =  np.zeros(frame1.shape, dtype=np.uint16)
        for i in aa:
            frame1_sgm  +=  (frame1 == circ[1, i]).astype(np.uint16)

        lbl       =  label(frame1_sgm, connectivity=1).astype(np.uint16)                # labeling of all the correctly segmented nuclei
        left      =  (np.sign(frame1) - np.sign(frame1_sgm)).astype(bool)                    # isolating the bad segmented nuclei
        left_lbl  =  label(left, connectivity=1).astype(np.uint16)                      # labeling the bad segmented nuclei
        left      =  np.sign(left_lbl).astype(bool)
        left_er   =  ndimage.morphology.binary_erosion(left)                                # binary erosion of the bad segmented nuclei (erosion is useful the make easier the task of the watershed)

        distance      =  ndimage.distance_transform_edt(left_er)                                                                        # watershe implementation
        lcl_max_crds    =  peak_local_max(np.copy(distance), footprint=np.ones((lmp, lmp), dtype=int))
        local_maxi  =  np.zeros(left_er.shape)
        local_maxi[lcl_max_crds[:, 0], lcl_max_crds[:, 1]]  =  1

        local_mx      =  filters.gaussian(local_maxi, 1) > 0
        local_mx_lbl  =  label(local_mx, connectivity=1)
        local_mx_rgp  =  regionprops(local_mx_lbl)
        ctrs_mx       =  np.zeros(frame1.shape, dtype=np.uint16)

        for i in range(len(local_mx_rgp)):
            ctrs_mx[np.round(local_mx_rgp[i]['Centroid'][0]).astype(int), np.round(local_mx_rgp[i]['Centroid'][1]).astype(int)]  =  1

        markers  =  ndimage.label(ctrs_mx)[0].astype(np.uint16)                                   # new labels after watershed
        labels   =  watershed(-distance, markers, mask=np.sign(left_lbl)).astype(np.uint16)

        lbl_fin  =  lbl + (lbl.max() + 1) * np.sign(labels) + labels            # add new labels to the final matrix evoiding to have the same label for some nuclei
        lbl_fin  =  label(lbl_fin, connectivity=1).astype(np.uint16)        # relabeling of the new matrix

        left_left  =  label(np.sign(frame1) - np.sign(lbl_fin), connectivity=1).astype(np.uint16)      # if there are still left labels, they are add to final labeled matrix, eventual corrections will be left to the user for manual corrections

        lbl_fin  =  lbl_fin + (lbl_fin.max() + 1) * np.sign(left_left) + left_left
        lbl_fin  =  remove_small_objects(lbl_fin, 60).astype(np.uint16)

        lbl_fin  =  label(lbl_fin, connectivity=1).astype(np.uint16)

        self.lbl_fin  =  lbl_fin


