"""This function generates an xls file with all the activation information.

Starting from the false colored time series we follow the activation history of
every single nucleus. For the nuclei in the after-mitosis part, we use the label of
the tracked nuclei (done not considering mothers and daughters, everyone by itself)
to give the 'daughter label'. If some nucleus is activated during the mitosis, we use
the DuringMitosisDaughterLabel.
"""



import datetime
import numpy as np
import xlwt
import tifffile
from skimage.measure import regionprops
from skimage.color import label2rgb
from PyQt5 import QtWidgets

import MitosisFrameFinder
import SpotsConnection



class WriteMergedData:
    """This class writes the tiff file of traked nuclei and false colored"""
    def __init__(self, nuc_act3c, first, foldername):

        tifffile.imwrite(foldername + '/false_colored.tif', np.rot90(nuc_act3c[:, :, ::-1, :], axes=(1, 2)).astype("uint8"))

        mycmap   =  np.fromfile('mycmap.bin', 'uint16').reshape((10000, 3))
        first3c  =  label2rgb(first, bg_color=[0, 0, 0], bg_label=0, colors=mycmap)

        tifffile.imwrite(foldername + '/tracked_nuclei.tif', np.rot90(first3c[:, :, ::-1, :], axes=(1, 2)).astype("uint8"))



class WriteMergedDataSpatial:
    """Write the spatial information into excell file"""
    def __init__(self, folder_path, nuclei_active, nuc_wild, nuc_track, y_min, y_max, foldername, mask, spots_tracked, ints_tot, vol_tot, soft_version):

        mit_bound  =  np.load(folder_path + '/mtss_boundaries.npy')
        frames_bm  =  mit_bound[0]
        frames_dm  =  mit_bound[1] - mit_bound[0]

        spots_special  =  SpotsConnection.SpotsConnection(nuc_wild[frames_bm + frames_dm - 1:], np.sign(spots_tracked[frames_bm + frames_dm - 1:]), 5).spots_tracked   # spots associated to the wild nuclei (we have a different label for spots associated to daughters)
        spots_special  =  np.concatenate((np.zeros((frames_bm + frames_dm - 1, spots_special.shape[1], spots_special.shape[2]), dtype=np.uint16), spots_special), axis=0)          # padding with zeros to obtain the same shape

        msk_tot      =  (nuclei_active == 2).astype(np.uint16)                                             # activated nuclei have label 2
        idx_sp       =  msk_tot * nuc_track                                                             # list of labels - trick to make it faster
        idx_sp       =  idx_sp[idx_sp != 0]
        idx_sp       =  np.unique(idx_sp)

        steps        =  nuc_wild.shape[0]
        presmtx_in   =  np.zeros((steps, idx_sp.size, 2), dtype=np.bool)                                # presence matrix (time steps x number of spots x 2-for daughters) of internal nuclei
        presmtx_ex   =  np.zeros((steps, idx_sp.size, 2), dtype=np.bool)                                # presence matrix (time steps x number of spots x 2-for daughters) of external nuclei
        presmtx_00   =  np.zeros((steps, idx_sp.size, 2), dtype=np.bool)                                # presence matrix (time steps x number of spots x 2-for daughters) of nuclei who's daughter are one internal and the other external
        g_dist_tot1  =  np.zeros(idx_sp.size)                                                           # vector with the average distance from gastrulation for daughter 1 including 13th cycle
        g_dist_am1   =  np.zeros(idx_sp.size)                                                           # vector with the average distance from gastrulation for daughter 1 excluding 13th cycle
        g_dist_tot2  =  np.zeros(idx_sp.size)                                                           # vector with the average distance from gastrulation for daughter 2 including 13th cycle
        g_dist_am2   =  np.zeros(idx_sp.size)                                                           # vector with the average distance from gastrulation for daughter 2 excluding 13th cycle
        mts_frm      =  np.zeros(idx_sp.size, dtype=np.uint16)                                          # vector of the mitosis frame number
        g_line_y     =  (y_max - y_min) / 2 + y_min                                                     # y coordinate of the gastrulation line

        book6    =  xlwt.Workbook(encoding='utf-8')
        sheet61  =  book6.add_sheet("Mother Inact-Ints")
        sheet62  =  book6.add_sheet("Mother Act-Ints")
        sheet63  =  book6.add_sheet("Mother Inact-Vols")
        sheet64  =  book6.add_sheet("Mother Act-Vols")

        sheet61.write(0, 0, "Time")
        sheet62.write(0, 0, "Time")
        sheet63.write(0, 0, "Time")
        sheet64.write(0, 0, "Time")
        [sheet61.write(1 + t, 0, str(t)) for t in range(steps)]
        [sheet62.write(1 + t, 0, str(t)) for t in range(steps)]
        [sheet63.write(1 + t, 0, str(t)) for t in range(steps)]
        [sheet64.write(1 + t, 0, str(t)) for t in range(steps)]

        pbar  =  ProgressBar(total1=idx_sp.size)
        pbar.show()

        k_13     =  0
        k_24     =  0
        flag_ia  =  "i"
        for k in range(idx_sp.size):
            pbar.update_progressbar(k)
#             print(k)
            spt_mtx     =  (spots_tracked == idx_sp[k])
            smpl        =  (nuc_track == idx_sp[k]) * nuc_wild                                                  # goes on tracking up to the moment the nucleus spilt (if both daughters remain in the view field)
            mts_ref     =  MitosisFrameFinder.MitosisFrameFinder(smpl).mpts
            mts_frm[k]  =  mts_ref[1]
            nu          =  np.unique(smpl[frames_bm + frames_dm + 1])[1:]                                       # find the tags of the daughters


            if np.sum(spt_mtx[:mts_frm[k]]) == 0:                                                               # translate the mather inactive condition
                sheet61.write(0, 1 + 2 * k_13, "Nuc_" + str(idx_sp[k]))
                sheet63.write(0, 1 + 2 * k_13, "Nuc_" + str(idx_sp[k]))
                [sheet61.write(1 + ii, 1 + 2 * k_13, 0) for ii in range(mts_frm[k])]
                [sheet63.write(1 + ii, 1 + 2 * k_13, 0) for ii in range(mts_frm[k])]
                flag_ia   =  "i"
            else:                                                                                               # mother active
                sheet62.write(0, 1 + 2 * k_24, "Nuc_" + str(idx_sp[k]))
                sheet64.write(0, 1 + 2 * k_24, "Nuc_" + str(idx_sp[k]))
                ints_prof  =  np.sum(ints_tot[:mts_frm[k]] * spt_mtx[:mts_frm[k]], axis=(1, 2))
                vols_prof  =  np.sum(vol_tot[:mts_frm[k]] * spt_mtx[:mts_frm[k]], axis=(1, 2))
                [sheet62.write(1 + ii, 1 + 2 * k_24, int(ints_prof[ii])) for ii in range(mts_frm[k])]           # write intensity and volume time series of the mother
                [sheet64.write(1 + ii, 1 + 2 * k_24, int(vols_prof[ii])) for ii in range(mts_frm[k])]
                flag_ia   =  "a"

            t_mtss    =  mts_ref[0]
            act_sing  =  np.sign(smpl).astype(np.uint16) + msk_tot                                              # puts the label 2 on the active nucleus, you can recognize the frame
            act_frms  =  np.where(act_sing.max(2).max(1) == 2)[0]
            lbls      =  np.zeros((steps, 2))

            for t in range(act_frms.size):
                lbl                           =  np.unique((act_sing[act_frms[t], :, :] == 2) * nuc_wild[act_frms[t], :, :])[1:]              # recognizes if the active nucleus has a sister
                lbls[act_frms[t], :lbl.size]  =  lbl

            profile  =  np.unique(lbls[mts_frm[k].astype(np.int):, :])[1:]
            lbls_xl  =  np.zeros(lbls.shape)

            if profile.size == 2:                                                                                       # if the nucleus has a sister in the view field the activation is recognized separately for both daughters
                lbl0                 =  nu[0]
                lbl1                 =  nu[1]
                lbls_xl[:t_mtss, 0]  =  np.sign(lbls[:t_mtss, 0])
                lbls_xl[t_mtss:, 0]  =  (lbls[t_mtss:, :] == lbl0).astype(np.uint16).sum(1)
                lbls_xl[t_mtss:, 1]  =  (lbls[t_mtss:, :] == lbl1).astype(np.uint16).sum(1)

            d1_ints  =  np.sum((spots_special == nu[0]) * ints_tot, axis=(1, 2))
            d2_ints  =  np.sum((spots_special == nu[1]) * ints_tot, axis=(1, 2))
            d1_vols  =  np.sum((spots_special == nu[0]) * vol_tot, axis=(1, 2))
            d2_vols  =  np.sum((spots_special == nu[1]) * vol_tot, axis=(1, 2))

            d1_t  =  np.where(d1_ints != 0)[0]                          # dX_t is the time of the first activation for a daughter: if it is not activate, put high number to write it second in the excel
            if d1_t.size > 0:
                d1_t  =  d1_t[0]
            else:
                d1_t  =  steps + 10

            d2_t  =  np.where(d2_ints != 0)[0]
            if d2_t.size > 0:
                d2_t  =  d2_t[0]
            else:
                d2_t  =  steps + 10

            if d1_t <= d2_t:                                                                                                        # id d1 activetrs before d2
                if flag_ia == "i":                                                                                                  # if mother inactive.... all the possible combinations follow
                    [sheet61.write(mts_frm[k] + 1 + ll, 1 + 2 * k_13, int(d1_ints[ll + mts_frm[k]])) for ll in range(steps - mts_frm[k])]
                    [sheet61.write(mts_frm[k] + 1 + ll, 2 + 2 * k_13, int(d2_ints[ll + mts_frm[k]])) for ll in range(steps - mts_frm[k])]
                    [sheet63.write(mts_frm[k] + 1 + ll, 1 + 2 * k_13, int(d1_vols[ll + mts_frm[k]])) for ll in range(steps - mts_frm[k])]
                    [sheet63.write(mts_frm[k] + 1 + ll, 2 + 2 * k_13, int(d2_vols[ll + mts_frm[k]])) for ll in range(steps - mts_frm[k])]
                    k_13  +=  1

                else:
                    [sheet62.write(mts_frm[k] + 1 + ll, 1 + 2 * k_24, int(d1_ints[ll + mts_frm[k]])) for ll in range(steps - mts_frm[k])]
                    [sheet62.write(mts_frm[k] + 1 + ll, 2 + 2 * k_24, int(d2_ints[ll + mts_frm[k]])) for ll in range(steps - mts_frm[k])]
                    [sheet64.write(mts_frm[k] + 1 + ll, 1 + 2 * k_24, int(d1_vols[ll + mts_frm[k]])) for ll in range(steps - mts_frm[k])]
                    [sheet64.write(mts_frm[k] + 1 + ll, 2 + 2 * k_24, int(d2_vols[ll + mts_frm[k]])) for ll in range(steps - mts_frm[k])]
                    k_24  +=  1

            if d1_t > d2_t:
                if flag_ia == "i":
                    [sheet61.write(mts_frm[k] + 1 + ll, 1 + 2 * k_13, int(d2_ints[ll + mts_frm[k]])) for ll in range(steps - mts_frm[k])]
                    [sheet61.write(mts_frm[k] + 1 + ll, 2 + 2 * k_13, int(d1_ints[ll + mts_frm[k]])) for ll in range(steps - mts_frm[k])]
                    [sheet63.write(mts_frm[k] + 1 + ll, 1 + 2 * k_13, int(d2_vols[ll + mts_frm[k]])) for ll in range(steps - mts_frm[k])]
                    [sheet63.write(mts_frm[k] + 1 + ll, 2 + 2 * k_13, int(d1_vols[ll + mts_frm[k]])) for ll in range(steps - mts_frm[k])]
                    k_13  +=  1
                else:
                    [sheet62.write(mts_frm[k] + 1 + ll, 1 + 2 * k_24, int(d2_ints[ll + mts_frm[k]])) for ll in range(steps - mts_frm[k])]
                    [sheet62.write(mts_frm[k] + 1 + ll, 2 + 2 * k_24, int(d1_ints[ll + mts_frm[k]])) for ll in range(steps - mts_frm[k])]
                    [sheet64.write(mts_frm[k] + 1 + ll, 1 + 2 * k_24, int(d2_vols[ll + mts_frm[k]])) for ll in range(steps - mts_frm[k])]
                    [sheet64.write(mts_frm[k] + 1 + ll, 2 + 2 * k_24, int(d1_vols[ll + mts_frm[k]])) for ll in range(steps - mts_frm[k])]
                    k_24  +=  1

            if profile.size == 1:                                                                                       # no sister, simple to recognize
                lbls_xl  =  np.sign(lbls)

            ctrs1  =  np.zeros(steps)
            ctrs2  =  np.zeros(steps)

            for t in range(t_mtss):
                rgp       =  regionprops(smpl[t, :, :].astype(np.int))
                ctrs1[t]  =  rgp[0]['centroid'][1]
                ctrs2[t]  =  rgp[0]['centroid'][1]
            for t in range(t_mtss, steps):
                rgp       =  regionprops(smpl[t, :, :].astype(np.int))       # if after the activation one of the daughters goes out of the view filed, we don't collect anymore info on their positions
                if len(rgp) > 1:
                    ctrs1[t]  =  rgp[0]['centroid'][1]
                    ctrs2[t]  =  rgp[1]['centroid'][1]

            g_dist_tot1[k]  =  np.abs(g_line_y - ctrs1.mean())
            g_dist_tot2[k]  =  np.abs(g_line_y - ctrs2.mean())
            g_dist_am1[k]   =  np.abs(g_line_y - ctrs1[t_mtss:].mean())
            g_dist_am2[k]   =  np.abs(g_line_y - ctrs2[t_mtss:].mean())

            a1, a2  =  0, 0                                                  # a1 and a2 are flags: a1 = 1 if internal and viceversa (the same for a2)
            if y_min < ctrs1.mean() < y_max:                                 # if both daughters are internal or external, they are written in the proper sheet.
                a1  =  1                                                     # If one is internal and the other is external, both are discarded by the excel

            if y_min < ctrs2.mean() < y_max:
                a2  =  1

            if a1 + a2 == 2:
                presmtx_in[:, k, :]  =  lbls_xl

            if a1 + a2 == 0:
                presmtx_ex[:, k, :]  =  lbls_xl

            if a1 + a2 == 1:
                presmtx_00[:, k, :]  =  lbls_xl

        mts_frm  =  mts_frm.astype(np.int)

        k_step       =  0
        presmtx_tot  =  presmtx_ex + presmtx_in + presmtx_00

        book6.save(foldername + "/MemorySpotsIntensityVolume" + "_" + folder_path[len(folder_path) - folder_path[::-1].find("/"):] + "find.xls")

        book5   =  xlwt.Workbook(encoding='utf-8')
        sheet5  =  book5.add_sheet("Sheet 1")

        sheet5.write(0, 0, "Nuc_id")
        sheet5.write(0, 1, "NC13_1stAct")
        sheet5.write(0, 2, "NC14_1stAct_D1")
        sheet5.write(0, 3, "NC14_1stAct_D2")
        sheet5.write(0, 4, "MTS_Frame")
        sheet5.write(0, 5, "AvTot_dist_D1")
        sheet5.write(0, 6, "AvAM_dist_D1")
        sheet5.write(0, 7, "AvTot_dist_D2")
        sheet5.write(0, 8, "AvAM_dist_D2")


        k_step  =  0
        for k in range(idx_sp.size):
            if presmtx_tot[:, k, :].sum() > 0:
                sheet5.write(k_step + 1, 0, "Nuc" + str(np.int(idx_sp[k])))

                a_bff  =  np.where(presmtx_tot[:mts_frm[k], k, :] != 0)[0]
                if len(a_bff) > 0:
                    sheet5.write(k_step + 1, 1, np.int(a_bff[0]) + 1)
                else:
                    sheet5.write(k_step + 1, 1, 0)

                a_bff  =  np.where(presmtx_tot[mts_frm[k]:, k, 0] != 0)[0]
                if len(a_bff) > 0:
                    alfa  =  np.int(a_bff[0] + mts_frm[k])
#                    sheet5.write(k_step + 1, 2, np.int(a_bff[0] + mts_frm[k]))
                else:
#                    sheet5.write(k_step + 1, 2, 0)
                    alfa  =  0

                a_bff  =  np.where(presmtx_tot[mts_frm[k]:, k, 1] != 0)[0]

                if len(a_bff) > 0:
                    print("Hello")
                    beta  =  np.int(a_bff[0] + mts_frm[k])
#                    sheet5.write(k_step + 1, 3, np.int(a_bff[0] + mts_frm[k]))
                else:
                    beta  =  0
#                    sheet5.write(k_step + 1, 3, 0)

                if (alfa != 0 and alfa < beta) or beta == 0:
                    sheet5.write(k_step + 1, 2, alfa)
                    sheet5.write(k_step + 1, 3, beta)
                    sheet5.write(k_step + 1, 5, np.int(g_dist_tot1[k]))
                    sheet5.write(k_step + 1, 6, np.int(g_dist_am1[k]))
                    sheet5.write(k_step + 1, 7, np.int(g_dist_tot2[k]))
                    sheet5.write(k_step + 1, 8, np.int(g_dist_am2[k]))
                else:
                    sheet5.write(k_step + 1, 3, alfa)
                    sheet5.write(k_step + 1, 2, beta)
                    sheet5.write(k_step + 1, 6, np.int(g_dist_tot1[k]))
                    sheet5.write(k_step + 1, 5, np.int(g_dist_am1[k]))
                    sheet5.write(k_step + 1, 8, np.int(g_dist_tot2[k]))
                    sheet5.write(k_step + 1, 7, np.int(g_dist_am2[k]))

                sheet5.write(k_step + 1, 4, np.int(mts_frm[k]))

#                sheet5.write(k_step + 1, 5, np.int(g_dist_tot1[k]))
#                sheet5.write(k_step + 1, 6, np.int(g_dist_am1[k]))
#                sheet5.write(k_step + 1, 7, np.int(g_dist_tot2[k]))
#                sheet5.write(k_step + 1, 8, np.int(g_dist_am2[k]))
                k_step  +=  1


        sheet5.write(5 + k_step, 0, foldername)
        sheet5.write(6 + k_step, 0, "Date")
        sheet5.write(6 + k_step, 1, datetime.datetime.now().strftime("%d-%b-%Y"))
        sheet5.write(7 + k_step, 0, "Software Version")
        sheet5.write(7 + k_step, 1, soft_version)
        book5.save(foldername + "/MemoryStudyWithDistances" + "_" + folder_path[len(folder_path) - folder_path[::-1].find("/"):] + ".xls")

        pbar.close()

        tifffile.imwrite(foldername + '/false_colored_spatial_constrain.tif', np.rot90(mask[:, :, ::-1, :], axes=(1, 2)).astype("uint8"))



class ProgressBar(QtWidgets.QWidget):
    """Simple progress bar widget"""
    def __init__(self, parent=None, total1=20):
        super(ProgressBar, self).__init__(parent)
        self.name_line1  =  QtWidgets.QLineEdit()

        self.progressbar  =  QtWidgets.QProgressBar()
        self.progressbar.setMinimum(1)
        self.progressbar.setMaximum(total1)


        main_layout  =  QtWidgets.QGridLayout()
        main_layout.addWidget(self.progressbar, 0, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Progress")
        self.setGeometry(500, 300, 300, 50)


    def update_progressbar(self, val1):
        """Progress bar updater"""
        self.progressbar.setValue(val1)
        QtWidgets.qApp.processEvents()


