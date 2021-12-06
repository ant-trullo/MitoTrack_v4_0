"""During mitosis nuclei have only one label (they miss the daughter label).

With this routine we assign the daughter label during mitosis. It needs to be
combined with the labels of the after mitosis part.
"""

import numpy as np
from skimage.morphology import label, remove_small_objects
from skimage.measure import regionprops



class DuringMitosisDaughterLabel:
    """The only class, does all the job"""
    def __init__(self, n_track_dm, idx_ref, idxs, n_sg):

        steps    =  n_track_dm.shape[0]
        im_test  =  (n_track_dm == idxs[idx_ref]).astype(np.int)                                         # select the nucleus and its daughters

        #  The built-in function label gives the lowest value (1) to the object with the lowest x coordinate of its lefter corner.
        #  If during the mitosis daughters move horizontally, they will be correctly segmented by label, if they move vertically, they won't.
        #  To avoid this, we rotate the video (twice, in different ways) in order to have it horizontal and we go on in the same way.
        #  If the split appends on a oblique line, both the procedures will produce the same correct result. Double rotation is useful to avoid
        #  that between the original and the rotated image there is a syncrony in label changing (rare, but ONCE I had this case)

        im_lbls  =  np.zeros(im_test.shape, dtype=np.uint16)
        for t in range(steps):
            im_lbls[t]  =  label(im_test[t] * n_sg[t], connectivity=1)
            im_lbls[t]  =  remove_small_objects(im_lbls[t].astype(np.int), 3)


        lbls_prof  =  im_lbls.max(2).max(1)
        if lbls_prof[-1] > 1:
            o  =  np.where(lbls_prof == 2)[0][0]

            im_test2  =  np.transpose(im_test, axes=(0, 2, 1))                                                                  # transposed matrix
            n_sg2     =  np.transpose(n_sg, axes=(0, 2, 1))                                                                  # transposed matrix
            im_lbls2  =  np.zeros(im_test2.shape, dtype=np.uint16)
            for t in range(steps):                                                                                              # labeling of transposed matrix
                im_lbls2[t]  =  label(im_test2[t] * n_sg2[t], connectivity=1)
                im_lbls2[t]  =  remove_small_objects(im_lbls2[t].astype(np.int), 3)
            im_lbls2  =  np.transpose(im_lbls2, axes=(0, 2, 1))

            im_test3  =  np.transpose(im_test, axes=(0, 2, 1))[:, ::-1, :]                                                      # transposed mirrored matrix
            n_sg3     =  np.transpose(n_sg, axes=(0, 2, 1))[:, ::-1, :]                                                      # transposed mirrored matrix
            im_lbls3  =  np.zeros(im_test2.shape, dtype=np.int16)
            for t in range(steps):                                                                                               # labeling of transposed mirrored matrix
                im_lbls3[t]  =  label(im_test3[t] * n_sg3[t], connectivity=1)
                im_lbls3[t]  =  remove_small_objects(im_lbls3[t].astype(np.int), 3)
            im_lbls3  =  np.transpose(im_lbls3[:, ::-1, :], axes=(0, 2, 1))

            ctrs                     =   np.zeros((2, 2))

            if np.abs((im_lbls - im_lbls2).sum(2).sum(1)).sum() == 0 or np.abs((im_lbls - im_lbls3).sum(2).sum(1)).sum() == 0:
                im_lbls_fin  =  im_lbls                                                                                            # rotated and not rotated give the same result. It is ok
            else:                                                                                                                  # if the results are different, we check if the split is horizontal or vertical
                rgp         =  regionprops(im_lbls[o, :, :].astype(np.int))
                ctrs[0, :]  =  rgp[0]['centroid'][0], rgp[0]['centroid'][1]
                ctrs[1, :]  =  rgp[1]['centroid'][0], rgp[1]['centroid'][1]
                if np.abs(ctrs[0, 0] - ctrs[1, 0]) < np.abs(ctrs[0, 1] - ctrs[1, 1]):                                              # vertical split, the rotated case is correct
                    im_lbls_fin  =  im_lbls2
                else:
                    im_lbls_fin  =  im_lbls                                                                                        # horizontal split, the rotated case is not correct

        else:
            im_lbls_fin  =  im_lbls

        self.im_lbls_fin  =  im_lbls_fin


