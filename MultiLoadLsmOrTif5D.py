"""This function loads and concatenates .lsm or tif.filedata.

Taking filenames as input, the output are the concatenated matrices of the
maximum intensity projection of red and green channels. Matrices are also flipped
and rotate to have a visualization conform to ImageJ standards.
"""


import numpy as np

import LoadLsmOrTif5D


class MultiLoadLsmOrTif5D:
    """Only class, does all the job"""
    def __init__(self, fnames, nucs_spts_ch):

        mt_buff   =  LoadLsmOrTif5D.LoadLsmOrTif5D(str(fnames[0]), nucs_spts_ch)
        flag_cut  =  0

        if len(mt_buff.red_mtx.shape) == 2:                                         # if a file contains just one time frame, we add an empty frame in order to have a 2 frames matrix to avoid concatenation problems
            imarray_red    =  np.zeros(np.append(2, mt_buff.red_mtx.shape))
            imarray_green  =  np.zeros(np.append(2, mt_buff.red_mtx.shape))

            imarray_red[1, :, :]    =  mt_buff.red_mtx
            imarray_green[1, :, :]  =  mt_buff.green_mtx

            flag_cut  =  1                                                          # this flag is to remember to remove the first empty frame in case we add it

        else:
            imarray_red    =  mt_buff.red_mtx
            imarray_green  =  mt_buff.green_mtx


        if len(fnames) > 1:
            for s in range(1, len(fnames)):

                mt_buff        =  LoadLsmOrTif5D.LoadLsmOrTif5D(str(fnames[s]), nucs_spts_ch)
                if len(mt_buff.red_mtx.shape) == 2:                                                                                             # check if some of the following files have 1 time frame or more: in case you have one you cannot concatenate directly.
                    imarray_red              =  np.concatenate((imarray_red, np.zeros(np.append(2, mt_buff.red_mtx.shape))), axis=0)            # we concatenate a matrix with 2 time frames, in the first of this we put the single time frame matrix loaded from the file and then remove the last zero frame
                    imarray_green            =  np.concatenate((imarray_green, np.zeros(np.append(2, mt_buff.red_mtx.shape))), axis=0)
                    imarray_red[-2, :, :]    =  mt_buff.red_mtx
                    imarray_green[-2, :, :]  =  mt_buff.green_mtx
                    imarray_red              =  np.delete(imarray_red, -1, 0)
                    imarray_green            =  np.delete(imarray_green, -1, 0)

                else:
                    imarray_red    =  np.concatenate((imarray_red, mt_buff.red_mtx), axis=0)
                    imarray_green  =  np.concatenate((imarray_green, mt_buff.green_mtx), axis=0)

        if flag_cut == 1:                                                                                                   # correction if the case in line 21 happens
            imarray_red    =  np.delete(imarray_red, 0, axis=0)
            imarray_green  =  np.delete(imarray_green, 0, axis=0)

        imarray_red    =  np.rot90(imarray_red, axes=(1, 2))[:, ::-1, :]
        imarray_green  =  np.rot90(imarray_green, axes=(1, 2))[:, ::-1, :]

        self.imarray_red    =  imarray_red
        self.imarray_green  =  imarray_green









