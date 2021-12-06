"""This function takes the output of the SpotsTracker to remove or visualize what was tracked.

"""


import numpy as np
from skimage.measure import regionprops, label
from PyQt5 import QtWidgets
# import pyqtgraph as pg


class VisualTracked:
    """In order to visualize the result of the tracking without generating
    extra matrices (they can be very big), a 3D matrix is generated with
    tracked spots having tag 2 and untracked with tag 1"""
    def __init__(self, spots_coords, spts_trck_info, spots_vol, green4D_shape):


        steps, zlen, xlen, ylen  =  green4D_shape                                           # 4D raw data shape
        visual_tracked           =  np.zeros((steps, xlen, ylen), dtype=np.int16)           # initialized output matrix

        pbar  =  ProgressBar(total1=len(spts_trck_info))                                    # progress bar
        pbar.show()
        pbar.update_progressbar(0)

        for k in range(len(spts_trck_info)):
            # print(k)
            pbar.update_progressbar(k)
            for t in range(spts_trck_info[k].shape[0]):
                spots_frame_lbls  =  SpotsReconstruction3D(spots_coords, spts_trck_info[k][t][0], zlen, xlen, ylen).spots_frame_lbls        # 3D reconstruction of a single time frame using coordinates
                sing_lbl          =  spots_frame_lbls[spts_trck_info[k][t][1], spts_trck_info[k][t][2], spts_trck_info[k][t][3]]            # detecting tag of the considered spot by its centroid coordinates

                if sing_lbl != 0:                                                                                                           # control statement
                    visual_tracked[spts_trck_info[k][t][0]]  +=  np.sign(np.sum((spots_frame_lbls == sing_lbl), axis=0)) * 1                # the spot summed in z is added to the output

                else:                                                                                                                       # in case of strange shape spots, centroid is not inside the spot itself; search for the tag around the the centroid
                    sing_lbl  =  spots_frame_lbls[np.max([spts_trck_info[k][t][1] - 3, 0]):np.min([spts_trck_info[k][t][1] + 4, zlen]), np.max([spts_trck_info[k][t][2] - 4, 0]):np.min([spts_trck_info[k][t][2] + 5, xlen]), np.max([spts_trck_info[k][t][3] - 4, 0]):np.min([spts_trck_info[k][t][3] + 5, ylen])].max()
                    if sing_lbl != 0:
                        visual_tracked[spts_trck_info[k][t][0]]  +=  np.sign(np.sum((spots_frame_lbls == sing_lbl), axis=0)) * 1            # the spot summed in z is added to the output

        self.visual_tracked  =  visual_tracked + 1 * np.sign(spots_vol)

        pbar.close()



class SpotsReconstruction3D:
    """Given the spots_coords and the frame number this matrix reconstruct and labels the 3D spots in a particular frame"""
    def __init__(self, spots_coords, t, zlen, xlen, ylen):

        spots_frame      =  np.zeros((zlen, xlen, ylen))
        spots_coords_fr  =  spots_coords[spots_coords[:, 0] == t]                                  # select the coordinates of the spots in a specific time frame

        for k in range(spots_coords_fr.shape[0]):
            spots_frame[spots_coords_fr[k, 1], spots_coords_fr[k, 2], spots_coords_fr[k, 3]] = 1       # building the spots in the frame as a binary

        spots_frame_lbls  =  label(spots_frame)                                                        # labels to the binary

        self.spots_frame_lbls  =  spots_frame_lbls



class RemoveTracked:
    """Tracked spots are removed from the original matrix"""
    def __init__(self, spots_coords, spots_tzxy, spots_ints, spots_vol, spts_trck_info, green4D):

        zlen, xlen, ylen  =  green4D.shape[1:]

        pbar  =  ProgressBar(total1=len(spts_trck_info))
        pbar.show()
        pbar.update_progressbar(0)

        for k in range(len(spts_trck_info)):
            pbar.update_progressbar(k)
            for t in range(spts_trck_info[k].shape[0]):
                spots_frame_lbls  =  SpotsReconstruction3D(spots_coords, spts_trck_info[k][t][0], zlen, xlen, ylen).spots_frame_lbls              # 3D reconstruction of a single time frame using coordinates
                sing_lbl          =  spots_frame_lbls[spts_trck_info[k][t][1], spts_trck_info[k][t][2], spts_trck_info[k][t][3]]                  # detecting tag of the considered spot by its centroid coordinates

                if sing_lbl != 0:
                    spots_vol[spts_trck_info[k][t][0]]   -=  np.sum(spots_frame_lbls == sing_lbl, axis=0)                                                           # remove tracked spots from the volume matrix
                    spots_ints[spts_trck_info[k][t][0]]  -=  np.sum((spots_frame_lbls == sing_lbl) * green4D[spts_trck_info[k][t][0]].astype(np.uint16), axis=0)       # remove tracked spots from the intensity matrix
                    rgp_sing                              =  regionprops((spots_frame_lbls == sing_lbl) * 1)                                                        # regionprops of the deleted spot
                    tzxy2remov                            =  np.append(spts_trck_info[k][t][0], np.round(rgp_sing[0]['centroid']).astype(np.uint16))                   # centroid of the deleted spot
                    idx2remov                             =  np.where(np.sum(spots_tzxy == tzxy2remov, axis=1) == 4)[0][0]                                          # find the coordinate of the deleted spot in the centroid matrix
                    spots_tzxy                            =  np.delete(spots_tzxy, idx2remov, 0)                                                                    # remove the coordinates of the centroid of the tracked spot

                else:
                                                                                                                                                                    # in case of strange shape spots, centroid is not inside the spot itself; search for the tag around the the centroid
                    sing_lbl  =  spots_frame_lbls[np.max([spts_trck_info[k][t][1] - 2, 0]):np.min([spts_trck_info[k][t][1] + 3, zlen]), np.max([spts_trck_info[k][t][2] - 2, 0]):np.min([spts_trck_info[k][t][2] + 3, xlen]), np.max([spts_trck_info[k][t][3] - 2, 0]):np.min([spts_trck_info[k][t][3] + 3, ylen])].max()
                    if sing_lbl != 0:
                        spots_vol[spts_trck_info[k][t][0]]   -=  np.sum(spots_frame_lbls == sing_lbl, axis=0)
                        spots_ints[spts_trck_info[k][t][0]]  -=  np.sum((spots_frame_lbls == sing_lbl) * green4D[spts_trck_info[k][t][0]].astype(np.int), axis=0)
                        rgp_sing                              =  regionprops((spots_frame_lbls == sing_lbl) * 1)
                        tzxy2remov                            =  np.append(spts_trck_info[k][t][0], np.round(rgp_sing[0]['centroid']).astype(np.int))
                        idx2remov                             =  np.where(np.sum(spots_tzxy == tzxy2remov, axis=1) == 4)[0][0]
                        spots_tzxy                            =  np.delete(spots_tzxy, idx2remov, 0)

        self.spots_ints  =  spots_ints
        self.spots_vol   =  spots_vol
        self.spots_tzxy  =  spots_tzxy



class ProgressBar(QtWidgets.QWidget):
    """Simple progressbar widget"""
    def __init__(self, parent=None, total1=20):
        super(ProgressBar, self).__init__(parent)
        self.name_line1  =  QtWidgets.QLineEdit()

        self.progressbar1  =  QtWidgets.QProgressBar()
        self.progressbar1.setMinimum(1)
        self.progressbar1.setMaximum(total1)

        main_layout  =  QtWidgets.QGridLayout()
        main_layout.addWidget(self.progressbar1, 0, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Progress")
        self.setGeometry(500, 300, 300, 50)

    def update_progressbar(self, val1):
        """Progress bar updater"""
        self.progressbar1.setValue(val1)
        QtWidgets.qApp.processEvents()





# ints for the removal of spts_coords values of the tracked spots
# ant = np.arange(33)
# ant = np.append(ant, np.array([9, 10, 11]))
# ant = ant.reshape((12, 3))
# ff = np.delete(ant, np.where((ant == (9, 10, 11)).all(axis=1))[0], axis=0)





# class RemoveTracked:
#    """Tracked spots are removed from the original matrix"""
#    def __init__(self, spots_lbls, spots_ints, spots_vol, spts_trck_info):

#        steps, zlen, xlen, ylen  =  spots_lbls.shape
       
#        pbar  =  TripleProgressBar(total1=len(spts_trck_info))
#        pbar.show()
# #        pbar_idx  =  0
#        pbar.update_progressbar1(0)

#        for k in range(len(spts_trck_info)):
#            pbar.update_progressbar1(k)
#            for t in range(spts_trck_info[k].shape[0]):
#                sing_lbl                             =  spots_lbls[spts_trck_info[k][t][0], spts_trck_info[k][t][1], spts_trck_info[k][t][2], spts_trck_info[k][t][3]]
#                if sing_lbl != 0:
#                    spots_lbls[spts_trck_info[k][t][0]]  =  np.where(spots_lbls[spts_trck_info[k][t][0]] == sing_lbl, 0, spots_lbls[spts_trck_info[k][t][0]])
#                else:
#                    sing_lbl  =  spots_lbls[np.max([spts_trck_info[k][t][0] - 1, 0]):np.min([spts_trck_info[k][t][0] + 2, steps]), np.max([spts_trck_info[k][t][1] - 1, 0]):np.min([spts_trck_info[k][t][1] + 2, zlen]), np.max([spts_trck_info[k][t][2] - 1, 0]):np.min([spts_trck_info[k][t][2] + 2, xlen]), np.max([spts_trck_info[k][t][3] - 1, 0]):np.min([spts_trck_info[k][t][3] + 2, ylen])].max()
#                    if sing_lbl != 0:
#                        spots_lbls[spts_trck_info[k][t][0]]  =  np.where(spots_lbls[spts_trck_info[k][t][0]] == sing_lbl, 0, spots_lbls[spts_trck_info[k][t][0]])
       
#        t_rgp_max  =  0
#        for j in range(len(spts_trck_info)):
#            t_rgp_max  =  np.max([t_rgp_max, spts_trck_info[j].shape[0]]) 

#        pbar.progressbar2.setMaximum(t_rgp_max)
#        for tt in range(t_rgp_max):
#            pbar.update_progressbar2(tt)
#            mask_bff         =  np.sign(spots_lbls[tt].sum(0))
#            spots_ints[tt]  *=  mask_bff
#            spots_vol[tt]   *=  mask_bff

#        spots_tzxy  =  []
#        steps       =  spots_ints.shape[0]
#        pbar.progressbar3.setMaximum(steps)
#        for t in range(steps):
#            pbar.update_progressbar3(t)
#            rgp_t  =  regionprops(spots_lbls[t])
#            for kk in range(len(rgp_t)):
#                spots_tzxy.append([t, np.round(np.asarray(rgp_t[kk]['centroid'])).astype(np.int)]) 

#        self.spots_lbls  =  spots_lbls
#        self.spots_ints  =  spots_ints
#        self.spots_vol   =  spots_vol
#        self.spots_tzxy  =  spots_tzxy
