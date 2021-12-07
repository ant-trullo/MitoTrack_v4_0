"""This is the main window of the softemory, MitoTrack.
This is version 4.0 since June 2020
"""

import os.path
import sys
import time
import traceback
from importlib import reload
import numpy as np
import pyqtgraph as pg
import skimage.morphology as skmr
from skimage.measure import regionprops
from scipy import ndimage
from PyQt5.QtCore import Qt
from PyQt5 import QtGui, QtWidgets, QtCore

import NucleiDetect
import NucleiDetectLog
import NucleiSegmentStackMultiCore
import NucleiConnectMultiCore
import NucleiSpotsConnection
import MultiLoadLsmOrTif5D
import MultiLoadCzi5D
import SpotsConnection
import LabelsModify
import MitosisCrossing3
import MergePartials
import NucleiConnect4MTool
import WriteMergedData
import EvolutionController
import FakeSpotsRemover
import FromTile2GlobCoordinate
import SaveMergedResults
import LoadMergedResults
import PopUpTool
import LoadPartialAnalysisTool
import SpotsDetectionChopper
import SpotTracker
import VisualTracked
import SpotsFeaturesSaver
import SpotsFeaturesLoader
import FalseColoredMakeUp


class MainWindow(QtWidgets.QMainWindow):
    """Main windows: coordinates all the actions, algorithms, visualization tools and analysis tools."""
    def __init__(self, parent=None):

        QtWidgets.QMainWindow.__init__(self, parent)

        widget = QtWidgets.QWidget(self)
        self.setCentralWidget(widget)

        exitAction  =  QtWidgets.QAction(QtGui.QIcon('Icons/exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)

        multiloadAction  =  QtWidgets.QAction(QtGui.QIcon('Icons/load-hi.png'), '&Load data file', self)
        multiloadAction.setShortcut('Ctrl+L')
        multiloadAction.setStatusTip('Load .tif, .lsm or .czi files: if they are more than one, they will be concatenate')
        multiloadAction.triggered.connect(self.load_several_files)

        load_partial_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/load-hi.png'), '&Load partial analysis', self)
        load_partial_action.setShortcut('Ctrl+P')
        load_partial_action.setStatusTip('Load partial analysis')
        load_partial_action.triggered.connect(self.load_partial_analysis)

        save_partial_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/save-md.png'), '&Save partial analysis', self)
        save_partial_action.setShortcut('Ctrl+S')
        save_partial_action.setStatusTip('Save partial analysis')
        save_partial_action.triggered.connect(self.save_partial)

        popup_nuclei_raw_action  =  QtWidgets.QAction('&Pop-up Raw Nuclei', self)
        popup_nuclei_raw_action.setStatusTip('Pop up image frames with the raw nuclei data')
        popup_nuclei_raw_action.triggered.connect(self.popup_nuclei_raw)

        popup_nuclei_detected_action  =  QtWidgets.QAction('&Pop-up Detected Nuclei', self)
        popup_nuclei_detected_action.setStatusTip('Pop up image window with the detected nuclei')
        popup_nuclei_detected_action.triggered.connect(self.popup_nuclei_detected)

        popup_nuclei_segmented_action  =  QtWidgets.QAction('&Pop-up Segmented Nuclei', self)
        popup_nuclei_segmented_action.setStatusTip('Pop up image window with the detected nuclei')
        popup_nuclei_segmented_action.triggered.connect(self.popup_nuclei_segmented)

        popup_nuclei_tracked_action  =  QtWidgets.QAction('&Pop-up Tracked Nuclei', self)
        popup_nuclei_tracked_action.setStatusTip('Pop up image window with the tracked nuclei')
        popup_nuclei_tracked_action.triggered.connect(self.popup_nuclei_trackeded)

        popup_spots_raw_action  =  QtWidgets.QAction('&Pop-up Raw Spots', self)
        popup_spots_raw_action.setStatusTip('Pop up image window with the raw spots')
        popup_spots_raw_action.triggered.connect(self.popup_spots_raw)

        popup_spots_segm_action  =  QtWidgets.QAction('&Pop-up Segmented Spots', self)
        popup_spots_segm_action.setStatusTip('Pop up image window with the segmented spots')
        popup_spots_segm_action.triggered.connect(self.popup_spots_segm)

        popup_nucactive_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/popup.png'), '&Pop-up Active Nuclei', self)
        popup_nucactive_action.setStatusTip('Pop up image window with the active nuclei map')
        popup_nucactive_action.triggered.connect(self.popup_nucactive)

        merge_analysis_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/merge.png'), '&Merge Partial Analysis', self)
        merge_analysis_action.setStatusTip('Merge the partial analysis and elaborate the 3 analysis')
        merge_analysis_action.triggered.connect(self.merge_analysis)
        merge_analysis_action.setShortcut('Ctrl+M')

        crop_stack_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/crop.png'), '&Crop Stack', self)
        crop_stack_action.setStatusTip('Crop data stack')
        crop_stack_action.triggered.connect(self.crop_stack)
        crop_stack_action.setShortcut('Ctrl+J')

        tile_coordinates_action  =  QtWidgets.QAction('&Tile Coordinates', self)
        tile_coordinates_action.setStatusTip('From tile image finds the coordinates relative to the nucleus of the analyzed stack')
        tile_coordinates_action.triggered.connect(self.tile_coordinates)
        tile_coordinates_action.setShortcut('Ctrl+T')

        load_merged_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/load-hi.png'), '&Load Merged Analysis', self)
        load_merged_action.setStatusTip('Load the merged analysis')
        load_merged_action.triggered.connect(self.load_merged)
        load_merged_action.setShortcut('Ctrl+K')

        set_color_channel_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/set_channels.png'), '&Set Channels', self)
        set_color_channel_action.setStatusTip('Set Channels Numbers for Loading')
        set_color_channel_action.triggered.connect(self.set_color_channel)
        set_color_channel_action.setShortcut('Ctrl+G')

        rmv_mitoticalTS_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/eraser.png'), '&Remove Mitotical TS', self)
        rmv_mitoticalTS_action.setStatusTip('Pop up tool to remove mitotical spots')
        rmv_mitoticalTS_action.triggered.connect(self.rmv_mitoticalTS)
        rmv_mitoticalTS_action.setShortcut('Ctrl+U')

        false_clc_makeup_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/make_up_transp.png'), '&False Colored Pretty', self)
        false_clc_makeup_action.setStatusTip('Generates multi tiff file of the false colored movie with nicer colors')
        false_clc_makeup_action.triggered.connect(self.false_clc_makeup)

        check_results_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/check_results.png'), '&Check Results', self)
        check_results_action.setStatusTip('Tool to check results of the analysis')
        check_results_action.triggered.connect(self.check_results)

        self.statusBar()

        menubar   =  self.menuBar()

        fileMenu  =  menubar.addMenu('&File')
        fileMenu.addAction(multiloadAction)
        fileMenu.addAction(save_partial_action)
        fileMenu.addAction(load_partial_action)
        fileMenu.addAction(load_merged_action)
        fileMenu.addAction(exitAction)

        modifyMenu  =  menubar.addMenu('&Modify')
        modifyMenu.addAction(crop_stack_action)
        modifyMenu.addAction(set_color_channel_action)
        modifyMenu.addAction(rmv_mitoticalTS_action)

        popupMenu     =  menubar.addMenu('&PopUp')
        popup_nuclei  =  popupMenu.addMenu(QtGui.QIcon('Icons/popup.png'), "PopUp Nuclei")
        popup_nuclei.addAction(popup_nuclei_raw_action)
        popup_nuclei.addAction(popup_nuclei_detected_action)
        popup_nuclei.addAction(popup_nuclei_segmented_action)
        popup_nuclei.addAction(popup_nuclei_tracked_action)

        popup_spots  =  popupMenu.addMenu(QtGui.QIcon('Icons/popup.png'), "PopUp Spots")
        popup_spots.addAction(popup_spots_raw_action)
        popup_spots.addAction(popup_spots_segm_action)
        popupMenu.addAction(popup_nucactive_action)

        postanalysisMenu  =  menubar.addMenu("&Post Analysis")
        postanalysisMenu.addAction(merge_analysis_action)
        postanalysisMenu.addAction(tile_coordinates_action)
        postanalysisMenu.addAction(false_clc_makeup_action)
        postanalysisMenu.addAction(check_results_action)


        frame1  =  pg.ImageView(self)
        frame1.ui.roiBtn.hide()
        frame1.ui.menuBtn.hide()

        frame2   =  pg.ImageView(self)
        frame2.ui.roiBtn.hide()
        frame2.ui.menuBtn.hide()
        frame2.ui.histogram.hide()

        frame3   =  pg.ImageView(self)
        frame3.ui.roiBtn.hide()
        frame3.ui.menuBtn.hide()

        frame4   =  pg.ImageView(self)
        frame4.ui.roiBtn.hide()
        frame4.ui.menuBtn.hide()
        frame4.ui.histogram.hide()

        fname_edt = QtWidgets.QLineEdit(self)
        fname_edt.setToolTip('Name of the file you are working on')

        busy_lbl  =  QtWidgets.QLabel("Ready")
        busy_lbl.setStyleSheet('color: green')

        busy_box  =  QtWidgets.QHBoxLayout()
        busy_box.addWidget(busy_lbl)
        busy_box.addStretch()

        sld1  =  QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        sld1.valueChanged.connect(self.sld1_update)

        start_cut_btn  =  QtWidgets.QPushButton("Start", self)
        start_cut_btn.clicked.connect(self.start_cut)
        start_cut_btn.setToolTip("Select the first frame to consider")
        start_cut_btn.setFixedSize(50, 25)

        end_cut_btn  =  QtWidgets.QPushButton("End", self)
        end_cut_btn.clicked.connect(self.end_cut)
        end_cut_btn.setToolTip("Select the last frame to consider")
        end_cut_btn.setFixedSize(50, 25)

        reload_cut_btn  =  QtWidgets.QPushButton("Reload", self)
        reload_cut_btn.clicked.connect(self.reload_files)
        reload_cut_btn.setToolTip("Reload selected files")
        reload_cut_btn.setFixedSize(110, 25)

        chop_btn  =  QtWidgets.QPushButton("Chop Stack", self)
        chop_btn.clicked.connect(self.chop)
        chop_btn.setToolTip("Chop the stack in 3 parts: before, during and after mitosis")
        chop_btn.setFixedSize(110, 25)

        sel_fstchop_btn  =  QtWidgets.QPushButton("B-M", self)
        sel_fstchop_btn.clicked.connect(self.sel_fstchop)
        sel_fstchop_btn.setToolTip("Select the stack chop before mitosis")
        sel_fstchop_btn.setFixedSize(33, 25)

        sel_scdchop_btn  =  QtWidgets.QPushButton("M", self)
        sel_scdchop_btn.clicked.connect(self.sel_scdchop)
        sel_scdchop_btn.setToolTip("Select the stack chop with mitosis")
        sel_scdchop_btn.setFixedSize(33, 25)

        sel_trdchop_btn  =  QtWidgets.QPushButton("A-M", self)
        sel_trdchop_btn.clicked.connect(self.sel_trdchop)
        sel_trdchop_btn.setToolTip('Select the stack chop after mitosis')
        sel_trdchop_btn.setFixedSize(33, 25)

        sel_lbl  =  QtWidgets.QLabel("  ", self)
        sel_lbl.setFixedSize(110, 25)

        nuc_detect_btn  =  QtWidgets.QPushButton("N-Detect", self)
        nuc_detect_btn.clicked.connect(self.nuclei_detection)
        nuc_detect_btn.setToolTip('Detect nuclei generating a B&W time series')
        nuc_detect_btn.setFixedSize(110, 25)

        parameter_detect_lbl  =  QtWidgets.QLabel('Gauss Size', self)
        parameter_detect_lbl.setFixedSize(60, 25)

        parameter_detect_edt  =  QtWidgets.QLineEdit(self)
        parameter_detect_edt.textChanged[str].connect(self.parameter_detect_var)
        parameter_detect_edt.setToolTip('Set the size of the Gaussian Filter for the pre-smoothing, or the thresholding coefficient (suggested value 1.5 for Gauss FLT and 1.0 for LogFlt)')
        parameter_detect_edt.setFixedSize(35, 25)

        nuc_segment_btn  =  QtWidgets.QPushButton("N-Segment", self)
        nuc_segment_btn.clicked.connect(self.nuclei_segmentation)
        nuc_segment_btn.setToolTip('Segment nuclei giving them a random color')
        nuc_segment_btn.setFixedSize(110, 25)

        gfilt_water_lbl  =  QtWidgets.QLabel('W-Shed', self)
        gfilt_water_lbl.setFixedSize(60, 25)

        gfilt_water_edt  =  QtWidgets.QLineEdit(self)
        gfilt_water_edt.textChanged[str].connect(self.gfilt_water_var)
        gfilt_water_edt.setToolTip('Set the size parameter for the Water Shed algorithm (suggested value 7)')
        gfilt_water_edt.setFixedSize(35, 25)

        circ_thr_lbl  =  QtWidgets.QLabel('Circ Thr', self)
        circ_thr_lbl.setFixedSize(60, 25)

        circ_thr_edt  =  QtWidgets.QLineEdit(self)
        circ_thr_edt.textChanged[str].connect(self.circ_thr_var)
        circ_thr_edt.setToolTip('Circularity Threshold of the detected nuclei (suggested value is 0.65)')
        circ_thr_edt.setFixedSize(35, 25)

        modify_segm_btn  =  QtWidgets.QPushButton("Modify Segm", self)
        modify_segm_btn.clicked.connect(self.modify_cycle_tool)
        modify_segm_btn.setToolTip('Activate a tool to modify Nuclei Segmentation')
        modify_segm_btn.setFixedSize(110, 25)

        gfilt2_detect_lbl  =  QtWidgets.QLabel('Gauss Size', self)
        gfilt2_detect_lbl.setFixedSize(60, 25)
        gfilt2_detect_lbl.setEnabled(False)

        gfilt2_detect_edt  =  QtWidgets.QLineEdit(self)
        gfilt2_detect_edt.textChanged[str].connect(self.gfilt2_detect_var)
        gfilt2_detect_edt.setToolTip('Sets the size of the Gaussian Filter for the pre-smoothing')
        gfilt2_detect_edt.setFixedSize(35, 25)
        gfilt2_detect_edt.setEnabled(False)

        mdetect_btn  =  QtWidgets.QPushButton("M-Detect", self)
        mdetect_btn.clicked.connect(self.mdetect)
        mdetect_btn.setToolTip('Detect Nuclei During Mitosis')
        mdetect_btn.setFixedSize(110, 25)
        mdetect_btn.setEnabled(False)

        mdetect_modify_btn  =  QtWidgets.QPushButton("Modify M-Track", self)
        mdetect_modify_btn.clicked.connect(self.modify_mitosis_tool)
        mdetect_modify_btn.setToolTip('Activate a tool to modify Tracking Across Mitosis')
        mdetect_modify_btn.setFixedSize(110, 25)
        mdetect_modify_btn.setEnabled(False)

        nuc_track_btn  =  QtWidgets.QPushButton("N-Track", self)
        nuc_track_btn.clicked.connect(self.nuclei_tracking)
        nuc_track_btn.setToolTip('Track segmented nuclei')
        nuc_track_btn.setFixedSize(110, 25)

        dist_thr_lbl  =  QtWidgets.QLabel('Dist Thr', self)
        dist_thr_lbl.setFixedSize(60, 25)

        dist_thr_edt  =  QtWidgets.QLineEdit(self)
        dist_thr_edt.textChanged[str].connect(self.dist_thr_var)
        dist_thr_edt.setToolTip('Distance threshold to track nuclei (suggested value 10)')
        dist_thr_edt.setFixedSize(35, 25)

        spots_thr_lbl  =  QtWidgets.QLabel('Spots Thr', self)
        spots_thr_lbl.setFixedSize(60, 25)

        spots_thr_edt  =  QtWidgets.QLineEdit(self)
        spots_thr_edt.textChanged[str].connect(self.spots_thr_var)
        spots_thr_edt.setToolTip('Intensity threshold to segment spots: it is expressed in terms of standard deviation (suggested value 7)')
        spots_thr_edt.setFixedSize(35, 25)

        spots_detect_btn  =  QtWidgets.QPushButton("S-Detect", self)
        spots_detect_btn.clicked.connect(self.spots_detect)
        spots_detect_btn.setToolTip('Detect spots generating a B&W time series')
        spots_detect_btn.setFixedSize(110, 25)

        spots_visual_btn  =  QtWidgets.QPushButton("Visualize Spots", self)
        spots_visual_btn.clicked.connect(self.spots_visual)
        spots_visual_btn.setToolTip('Activate a tool to check spots detection')
        spots_visual_btn.setFixedSize(110, 25)

        time_step_lbl  =  QtWidgets.QLabel('T Step', self)
        time_step_lbl.setFixedSize(50, 25)

        time_step_edt  =  QtWidgets.QLineEdit(self)
        time_step_edt.textChanged[str].connect(self.time_step_var)
        time_step_edt.setToolTip('Duration in seconds of the time step')
        time_step_edt.setFixedSize(45, 25)
        time_step_edt.setText('0')

        time_lbl = QtWidgets.QLabel("time  " + '0', self)
        time_lbl.setFixedSize(110, 25)

        gaus_log_detect_combo  =  QtWidgets.QComboBox(self)
        gaus_log_detect_combo.addItem("Gauss Flt")
        gaus_log_detect_combo.addItem("Log Flt")
        gaus_log_detect_combo.setCurrentIndex(0)
        gaus_log_detect_combo.setToolTip('Switch between a linear nuclei detection and a logaritmic nuclei detection')
        gaus_log_detect_combo.activated[str].connect(self.gaus_log_detect)

        hor_line_one  =  QtWidgets.QFrame()
        hor_line_one.setFrameStyle(QtWidgets.QFrame.HLine)

        frame_numb_lbl = QtWidgets.QLabel("frame  " + '0', self)
        frame_numb_lbl.setFixedSize(110, 13)

        cut_box_h  =  QtWidgets.QHBoxLayout()
        cut_box_h.addWidget(start_cut_btn)
        cut_box_h.addWidget(end_cut_btn)

        cut_box_h2  =  QtWidgets.QHBoxLayout()
        cut_box_h2.addWidget(sel_fstchop_btn)
        cut_box_h2.addWidget(sel_scdchop_btn)
        cut_box_h2.addWidget(sel_trdchop_btn)

        cut_box  =  QtWidgets.QVBoxLayout()
        cut_box.addLayout(cut_box_h)
        cut_box.addWidget(reload_cut_btn)
        cut_box.addWidget(chop_btn)
        cut_box.addLayout(cut_box_h2)
        cut_box.addWidget(sel_lbl)

        h1box  =  QtWidgets.QHBoxLayout()
        h1box.addWidget(frame1)
        h1box.addWidget(frame2)

        h2box  = QtWidgets.QHBoxLayout()
        h2box.addWidget(frame3)
        h2box.addWidget(frame4)

        v2box  =  QtWidgets.QVBoxLayout()
        v2box.addWidget(fname_edt)
        v2box.addLayout(h1box)
        v2box.addLayout(h2box)
        v2box.addWidget(sld1)
        v2box.addLayout(busy_box)

        gdet_hor  =  QtWidgets.QHBoxLayout()
        gdet_hor.addWidget(parameter_detect_lbl)
        gdet_hor.addWidget(parameter_detect_edt)

        key_ver1  =  QtWidgets.QVBoxLayout()
        key_ver1.addLayout(gdet_hor)
        key_ver1.addWidget(gaus_log_detect_combo)
        key_ver1.addWidget(nuc_detect_btn)
        key_ver1.addStretch()

        gseg_hor  =  QtWidgets.QHBoxLayout()
        gseg_hor.addWidget(gfilt_water_lbl)
        gseg_hor.addWidget(gfilt_water_edt)

        circ_thr_hor  =  QtWidgets.QHBoxLayout()
        circ_thr_hor.addWidget(circ_thr_lbl)
        circ_thr_hor.addWidget(circ_thr_edt)

        key_ver2  =  QtWidgets.QVBoxLayout()
        key_ver2.addLayout(gseg_hor)
        key_ver2.addLayout(circ_thr_hor)
        key_ver2.addWidget(nuc_segment_btn)
        key_ver2.addStretch()

        dist_thr_hor  =  QtWidgets.QHBoxLayout()
        dist_thr_hor.addWidget(dist_thr_lbl)
        dist_thr_hor.addWidget(dist_thr_edt)

        key_modifing  =  QtWidgets.QVBoxLayout()
        key_modifing.addWidget(modify_segm_btn)

        gdet_hor2  =  QtWidgets.QHBoxLayout()
        gdet_hor2.addWidget(gfilt2_detect_lbl)
        gdet_hor2.addWidget(gfilt2_detect_edt)

        key_mdetect  =  QtWidgets.QVBoxLayout()
        key_mdetect.addWidget(mdetect_btn)
        key_mdetect.addWidget(mdetect_modify_btn)

        key_ver3  =  QtWidgets.QVBoxLayout()
        key_ver3.addLayout(dist_thr_hor)
        key_ver3.addWidget(nuc_track_btn)
        key_ver3.addStretch()

        spots_thr_hor  =  QtWidgets.QHBoxLayout()
        spots_thr_hor.addWidget(spots_thr_lbl)
        spots_thr_hor.addWidget(spots_thr_edt)

        volume_thr_lbl  =  QtWidgets.QLabel('Spots Vol', self)
        volume_thr_lbl.setFixedSize(60, 25)

        volume_thr_edt  =  QtWidgets.QLineEdit(self)
        volume_thr_edt.textChanged[str].connect(self.volume_thr_var)
        volume_thr_edt.setToolTip('Minimum volume of a spot (suggested value 5)')
        volume_thr_edt.setFixedSize(35, 25)

        volume_thr_hor  =  QtWidgets.QHBoxLayout()
        volume_thr_hor.addWidget(volume_thr_lbl)
        volume_thr_hor.addWidget(volume_thr_edt)

        key_ver4  =  QtWidgets.QVBoxLayout()
        key_ver4.addLayout(spots_thr_hor)
        key_ver4.addLayout(volume_thr_hor)
        key_ver4.addWidget(spots_detect_btn)
        key_ver4.addWidget(spots_visual_btn)

        key_time_step  =  QtWidgets.QHBoxLayout()
        key_time_step.addWidget(time_step_lbl)
        key_time_step.addWidget(time_step_edt)

        key_tot  =  QtWidgets.QVBoxLayout()
        key_tot.addLayout(cut_box)
        key_tot.addStretch()
        key_tot.addLayout(key_ver1)
        key_tot.addLayout(key_ver2)
        key_tot.addLayout(key_modifing)
        key_tot.addLayout(key_ver3)
        key_tot.addStretch()
        key_tot.addLayout(gdet_hor2)
        key_tot.addLayout(key_mdetect)
        key_tot.addStretch()
        key_tot.addLayout(key_ver4)
        key_tot.addStretch()
        key_tot.addWidget(hor_line_one)
        key_tot.addLayout(key_time_step)
        key_tot.addWidget(time_lbl)
        key_tot.addWidget(frame_numb_lbl)

        layout   =  QtWidgets.QHBoxLayout(widget)
        layout.addLayout(v2box)
        layout.addLayout(key_tot)

        mycmap  =  np.fromfile("mycmap.bin", "uint16").reshape((10000, 3))
        self.colors4map  =  []
        for k in range(mycmap.shape[0]):
            self.colors4map.append(mycmap[k, :])
        self.colors4map[0]  =  np.array([0, 0, 0])
        self.colors4map[1]  =  np.array([1, 1, 1])

        self.frame1  =  frame1
        self.frame2  =  frame2
        self.frame3  =  frame3
        self.frame4  =  frame4

        self.fname_edt              =  fname_edt
        self.sld1                   =  sld1
        self.mdetect_btn            =  mdetect_btn
        self.gfilt2_detect_lbl      =  gfilt2_detect_lbl
        self.gfilt2_detect_edt      =  gfilt2_detect_edt
        self.mdetect_modify_btn     =  mdetect_modify_btn
        self.parameter_detect_lbl   =  parameter_detect_lbl
        self.parameter_detect_edt   =  parameter_detect_edt
        self.nuc_segment_btn        =  nuc_segment_btn
        self.nuc_detect_btn         =  nuc_detect_btn
        self.gfilt_water_lbl        =  gfilt_water_lbl
        self.gfilt_water_edt        =  gfilt_water_edt
        self.circ_thr_lbl           =  circ_thr_lbl
        self.circ_thr_edt           =  circ_thr_edt
        self.modify_segm_btn        =  modify_segm_btn
        self.nuc_track_btn          =  nuc_track_btn
        self.dist_thr_lbl           =  dist_thr_lbl
        self.dist_thr_edt           =  dist_thr_edt
        self.spots_thr_edt          =  spots_thr_edt
        self.busy_lbl               =  busy_lbl
        self.time_step_edt          =  time_step_edt
        self.gaus_log_detect_combo  =  gaus_log_detect_combo
        self.gaus_log_detect_value  =  "Gauss Flt"
        self.volume_thr_edt         =  volume_thr_edt

        self.data_flag             =  0
        self.labbs_flag            =  0
        self.nuclei_flag           =  0
        self.nuclei_t_visual_flag  =  0
        self.spots_segm_flag       =  0
        self.start_cut_var         =  0
        self.end_cut_var           =  0
        self.bm_dm_am              =  0
        self.t_track_end_value     =  0
        self.nucs_spts_ch          =  np.array([1, 0])

        self.time_lbl        =  time_lbl
        self.frame_numb_lbl  =  frame_numb_lbl
        self.sel_lbl         =  sel_lbl
        self.soft_version    =  "MitoTrack_v4.0"

        self.setGeometry(100, 100, 1200, 800)
        self.setWindowTitle(self.soft_version)
        self.setWindowIcon(QtGui.QIcon('Icons/MLL_Logo2.png'))
        self.show()

    def closeEvent(self, event):
        "Close the GUI, asking confirmation"
        quit_msg  =  "Are you sure you want to exit the program?"
        reply     =  QtWidgets.QMessageBox.question(self, 'Message', quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def busy_indicator(self):
        """Write a red text (BUSY) as a label on the GUI (bottom left)"""
        self.busy_lbl.setText("Busy")
        self.busy_lbl.setStyleSheet('color: red')

    def ready_indicator(self):
        """Write a green text (READY) as a label on the GUI (bottom left)"""
        self.busy_lbl.setText("Ready")
        self.busy_lbl.setStyleSheet('color: green')

    def crop_stack(self):
        """Call a popup tool to crop the raw data before the analysis"""
        self.mpp7  =  CroppingTool(self.imarray_red, self.imarray_green)
        self.mpp7.show()
        self.mpp7.procStart.connect(self.crop_tool_sgnl)

    def crop_tool_sgnl(self, message):
        """Update the work of the CroppingTool on the main GUI"""
        pts  =  self.mpp7.roi.parentBounds()
        x0   =  np.round(np.max([0, pts.x()])).astype(int)
        y0   =  np.round(np.max([0, pts.y()])).astype(int)
        x1   =  np.round(np.min([pts.x() + pts.width(), self.imarray_red.shape[1]])).astype(int)
        y1   =  np.round(np.min([pts.y() + pts.height(), self.imarray_red.shape[2]])).astype(int)

        self.imarray_red    =  self.imarray_red[:, x0:x1, y0:y1]
        self.imarray_green  =  self.imarray_green[:, x0:x1, y0:y1]
        self.frame1.setImage(self.imarray_red[message, :, :])
        self.frame3.setImage(self.imarray_green[message, :, :])

        self.mpp7.close()

    def gaus_log_detect(self, text):
        """Manage the flags changed with the combobox for nuclei detection"""
        self.gaus_log_detect_value  =  text
        if text == "Gauss Flt":
            self.parameter_detect_lbl.setText("Gauss Size")
        else:
            self.parameter_detect_lbl.setText("Thr Coeff")

    def load_several_files(self):
        """Load and concatenate several files to analyze"""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            self.fnames  =  QtWidgets.QFileDialog.getOpenFileNames(None, "Select czi data files to concatenate...", filter="*.lsm *.czi *.tif")[0]

            if str(self.fnames[0])[-3:] == 'lsm' or str(self.fnames[0])[-3:] == 'tif':
                self.filedata  =  MultiLoadLsmOrTif5D.MultiLoadLsmOrTif5D(self.fnames, self.nucs_spts_ch)
            if str(self.fnames[0])[-3:] == 'czi':
                self.filedata  =  MultiLoadCzi5D.MultiLoadCzi5D(self.fnames, self.nucs_spts_ch)
                self.time_step_edt.setText(str(self.filedata.time_step_value))

            self.frame1.setImage(self.filedata.imarray_red[0, :, :])
            self.frame3.setImage(self.filedata.imarray_green[0, :, :])
            self.data_flag  =  1
            self.sld1.setMaximum(self.filedata.imarray_red[:, 0, 0].size - 1)
            self.sld1.setValue(0)
            joined_fnames  = ' '
            for s in range(len(self.fnames)):
                joined_fnames  +=  str(self.fnames[s]) +  ' ----- '

            self.fname_edt.setText(joined_fnames)
        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def load_partial_analysis(self):
        """Load partial analysis in GUI to let you work on or check it back"""
        self.folder_storedata  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the folder with the analyzed data"))
        self.fname_edt.setText(self.folder_storedata)
        self.time_step_edt.setText(str(np.load(self.folder_storedata + "/time_step_value.npy")[0]))

    def start_cut(self):
        """Define the first frame you will analyze"""
        self.filedata.imarray_red    =  self.filedata.imarray_red[self.sld1.value():, :, :]
        self.filedata.imarray_green  =  self.filedata.imarray_green[self.sld1.value():, :, :]
        self.filedata.green4D        =  self.filedata.green4D[self.sld1.value():, :, :, :]
        self.start_cut_var           =  self.sld1.value()
        self.sld1.setMaximum(self.filedata.imarray_red[:, 0, 0].size - 1)
        self.sld1.setValue(0)

    def end_cut(self):
        """Define the last frame you will analyze"""
        self.filedata.imarray_red    =  self.filedata.imarray_red[:self.sld1.value() + 1, :, :]
        self.filedata.imarray_green  =  self.filedata.imarray_green[:self.sld1.value() + 1, :, :]
        self.filedata.green4D        =  self.filedata.green4D[:self.sld1.value() + 1, :, :, :]
        self.end_cut_var             =  self.sld1.value()
        self.sld1.setMaximum(self.filedata.imarray_red[:, 0, 0].size - 1)
        self.sld1.setValue(self.filedata.imarray_red[:, 0, 0].size - 1)

    def reload_files(self):
        """Reload the same files you already loaded without select them back"""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            if str(self.fnames[0])[-3:] == 'lsm' or str(self.fnames[0])[-3:] == 'tif':
                self.filedata  =  MultiLoadLsmOrTif5D.MultiLoadLsmOrTif5D(self.fnames, self.nucs_spts_ch)
            if str(self.fnames[0])[-3:] == 'czi':
                self.filedata  =  MultiLoadCzi5D.MultiLoadCzi5D(self.fnames, self.nucs_spts_ch)
                self.time_step_edt.setText(str(self.filedata.time_step_value))

            self.frame1.setImage(self.filedata.imarray_red[0, :, :])
            self.frame3.setImage(self.filedata.imarray_green[0, :, :])
            self.data_flag = 1
            self.sld1.setMaximum(self.filedata.imarray_red[:, 0, 0].size - 1)
            self.sld1.setValue(0)

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def chop(self):
        """Call the ChopStack tool to define analysis mile stones"""
        self.folder_storedata  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the folder to store data"))
        self.chopw             =  ChopStack(self.filedata.imarray_red, self.filedata.imarray_green, self.filedata.green4D, self.time_step_value, self.nucs_spts_ch, self.folder_storedata)
        self.chopw.show()

    def sel_fstchop(self):
        """Select the first analysis chop (before mitosis)"""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:

            self.sel_lbl.setText("Before Mitosis")
            self.filedata  =  LoadPartialAnalysisTool.LoadPartialAnalysisFrstChop(self.folder_storedata)

            self.sld1.setMaximum(self.filedata.imarray_red[:, 0, 0].size - 1)
            self.sld1.setValue(0)
            self.frame1.setImage(self.filedata.imarray_red[0, :, :])
            self.frame3.setImage(self.filedata.imarray_green[0, :, :])
            self.data_flag        =  1
            self.spots_segm_flag  =  0
            self.bm_dm_am         =  1

            if os.path.isfile(self.folder_storedata + "/lbl_nuclei_bm.npz"):
                file  =  open(self.folder_storedata + '/before_mitosis_params.txt', "r")
                a     =  file.readlines()
                if a[0][:3] == "Gau":
                    self.gaus_log_detect_combo.setCurrentIndex(0)
                    self.parameter_detect_lbl.setText("Gauss Size")
                    self.gaus_log_detect_value  =  "Gauss Flt"
                    self.parameter_detect_edt.setText(a[0][30:-1])

                if a[0][:3] == "Log":
                    self.gaus_log_detect_combo.setCurrentIndex(1)
                    self.parameter_detect_lbl.setText("Thr Coeff")
                    self.gaus_log_detect_value  =  "Log Flt"
                    self.parameter_detect_edt.setText(a[0][38:-1])

                self.gfilt_water_edt.setText(a[1][22:-1])
                self.circ_thr_edt.setText(a[2][24:-1])
                self.dist_thr_edt.setText(a[3][21:-1])
                self.spots_thr_edt.setText(a[4][18:-1])
                self.volume_thr_edt.setText(a[5][19:-1])
                self.nuclei_t       =  np.load(self.folder_storedata + '/lbl_nuclei_bm.npz')
                self.nuclei_t       =  self.nuclei_t['arr_0']
                self.nuclei_labels  =  self.nuclei_t
                self.frame2.setImage(self.nuclei_t[0, :, :], levels=(0, self.nuclei_t.max()))
                self.mycmap  =  pg.ColorMap(np.linspace(0, 1, self.nuclei_t.max()), color=self.colors4map)
                self.frame2.setColorMap(self.mycmap)
                self.nuclei_t_visual_flag  =  1

                #                 self.spots_tracked  =  np.load(self.folder_storedata + '/trk_spots_bm.npy')
                self.spots_tracked  =  SpotsFeaturesLoader.SpotsFeaturesLoader(self.folder_storedata + '/trk_spots_bm_coords.npy').spts_mtx
                self.spots_3D       =  LoadPartialAnalysisTool.LoadSpots(self.folder_storedata, "b")
                self.frame4.setImage(np.sign(self.spots_tracked[0, :, :]))
                self.spots_segm_flag  =  1

            self.gfilt2_detect_lbl.setEnabled(False)
            self.gfilt2_detect_edt.setEnabled(False)
            self.mdetect_btn.setEnabled(False)
            self.mdetect_modify_btn.setEnabled(False)
            self.parameter_detect_lbl.setEnabled(True)
            self.parameter_detect_edt.setEnabled(True)
            self.nuc_segment_btn.setEnabled(True)
            self.nuc_detect_btn.setEnabled(True)
            self.gfilt_water_lbl.setEnabled(True)
            self.gfilt_water_edt.setEnabled(True)
            self.circ_thr_edt.setEnabled(True)
            self.circ_thr_lbl.setEnabled(True)
            self.modify_segm_btn.setEnabled(True)
            self.nuc_track_btn.setEnabled(True)
            self.dist_thr_lbl.setEnabled(True)
            self.dist_thr_edt.setEnabled(True)

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def sel_scdchop(self):
        """Select the second analysis chop (during mitosis)"""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            self.sel_lbl.setText("During Mitosis")
            self.filedata  =  LoadPartialAnalysisTool.LoadPartialAnalysisScndChop(self.folder_storedata)

            self.sld1.setMaximum(self.filedata.imarray_red[:, 0, 0].size - 1)
            self.sld1.setValue(0)
            self.frame1.setImage(self.filedata.imarray_red[0, :, :])
            self.frame3.setImage(self.filedata.imarray_green[0, :, :])
            self.data_flag        =  1
            self.spots_segm_flag  =  0
            self.bm_dm_am         =  2

            if os.path.isfile(self.folder_storedata + '/lbl_nuclei_dm.npz'):
                file  =  open(self.folder_storedata + '/during_mitosis_params.txt', "r")
                a     =  file.readlines()
                self.gfilt2_detect_edt.setText(a[0][30:-1])
                self.spots_thr_edt.setText(a[1][18:-1])
                self.volume_thr_edt.setText(a[7][19:-1])
                self.dm     =  np.load(self.folder_storedata + '/lbl_nuclei_dm.npz')
                self.dm     =  self.dm['arr_0']
                #                 self.labbs  =  NucleiDetect.NucleiDetect(self.filedata.imarray_red, self.gfilt2_detect_value).labbs
                self.labbs  =  np.load(self.folder_storedata + '/lbl_segm_dm.npz')['arr_0']
                self.mpp2   =  ModifierMitosisTool(self.filedata.imarray_red, self.labbs, self.dm.astype(int), 0)
                self.mpp2.show()
                self.mpp2.size_thr_edt.setText(a[2][17:-1])
                self.mpp2.overlap_ratio_edt.setText(a[3][16:-1])
                self.mpp2.prolif_ratio_edt.setText(a[4][22:-1])
                self.mpp2.dist_singtrck_edt.setText(a[5][21:-1])

                self.frame2.setImage(self.labbs[0, :, :])
                self.nuclei_flag  =  1
                self.mpp2.procStart.connect(self.sgnl_update_mitosis)
                self.labbs_flag            =  0
                self.nuclei_flag           =  0
                self.nuclei_t_visual_flag  =  1

                #                 self.spots_segm     =  np.load(self.folder_storedata + '/trk_spots_dm.npy')
                self.spots_segm     =  SpotsFeaturesLoader.SpotsFeaturesLoader(self.folder_storedata + '/trk_spots_dm_coords.npy').spts_mtx
                #                 self.spots_tracked  =  np.load(self.folder_storedata + '/trk_spots_dm.npy')
                self.spots_tracked  =  SpotsFeaturesLoader.SpotsFeaturesLoader(self.folder_storedata + '/trk_spots_dm_coords.npy').spts_mtx
                self.spots_3D       =  LoadPartialAnalysisTool.LoadSpots(self.folder_storedata, "d")
                self.frame4.setImage(np.sign(self.spots_tracked[0, :, :]))
                self.spots_segm_flag  =  1

            self.gfilt2_detect_lbl.setEnabled(True)
            self.gfilt2_detect_edt.setEnabled(True)
            self.mdetect_btn.setEnabled(True)
            self.mdetect_modify_btn.setEnabled(True)
            self.parameter_detect_lbl.setEnabled(False)
            self.parameter_detect_edt.setEnabled(False)
            self.nuc_segment_btn.setEnabled(False)
            self.nuc_detect_btn.setEnabled(False)
            self.gfilt_water_lbl.setEnabled(False)
            self.gfilt_water_edt.setEnabled(False)
            self.circ_thr_edt.setEnabled(False)
            self.circ_thr_lbl.setEnabled(False)
            self.modify_segm_btn.setEnabled(False)
            self.nuc_track_btn.setEnabled(False)
            self.dist_thr_lbl.setEnabled(False)
            self.dist_thr_edt.setEnabled(False)

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def sel_trdchop(self):
        """Select the third analysis chop (after mitosis)"""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            self.sel_lbl.setText("After Mitosis")
            self.filedata  =  LoadPartialAnalysisTool.LoadPartialAnalysisThrdChop(self.folder_storedata)
            print(self.filedata.green4D.shape)

            self.sld1.setMaximum(self.filedata.imarray_red[:, 0, 0].size - 1)
            self.sld1.setValue(0)
            self.frame1.setImage(self.filedata.imarray_red[0, :, :])
            self.frame3.setImage(self.filedata.imarray_green[0, :, :])
            self.data_flag        =  1
            self.spots_segm_flag  =  0
            self.bm_dm_am         =  3

            if os.path.isfile(self.folder_storedata + "/lbl_nuclei_am.npz"):
                file  =  open(self.folder_storedata + '/after_mitosis_params.txt', "r")
                a     =  file.readlines()
                if a[0][:3] == "Gau":
                    self.gaus_log_detect_combo.setCurrentIndex(0)
                    self.parameter_detect_lbl.setText("Gauss Size")
                    self.gaus_log_detect_value  =  "Gauss Flt"
                    self.parameter_detect_edt.setText(a[0][30:-1])

                if a[0][:3] == "Log":
                    self.gaus_log_detect_combo.setCurrentIndex(1)
                    self.parameter_detect_lbl.setText("Thr Coeff")
                    self.gaus_log_detect_value  =  "Log Flt"
                    self.parameter_detect_edt.setText(a[0][38:-1])

                self.gfilt_water_edt.setText(a[1][22:-1])
                self.circ_thr_edt.setText(a[2][24:-1])
                self.dist_thr_edt.setText(a[3][21:-1])
                self.spots_thr_edt.setText(a[4][18:-1])
                self.volume_thr_edt.setText(a[5][19:-1])
                self.nuclei_t       =  np.load(self.folder_storedata + '/lbl_nuclei_am.npz')
                self.nuclei_t       =  self.nuclei_t['arr_0']
                self.nuclei_labels  =  self.nuclei_t
                self.frame2.setImage(self.nuclei_t[0, :, :], levels=(0, self.nuclei_t.max()))
                self.mycmap  =  pg.ColorMap(np.linspace(0, 1, self.nuclei_t.max()), color=self.colors4map)
                self.frame2.setColorMap(self.mycmap)
                self.nuclei_t_visual_flag  =  1

                self.spots_tracked  =  SpotsFeaturesLoader.SpotsFeaturesLoader(self.folder_storedata + '/trk_spots_am_coords.npy').spts_mtx
                self.spots_3D       =  LoadPartialAnalysisTool.LoadSpots(self.folder_storedata, "a")
                self.frame4.setImage(np.sign(self.spots_tracked[0, :, :]))
                self.spots_segm_flag  =  1

            self.gfilt2_detect_lbl.setEnabled(False)
            self.gfilt2_detect_edt.setEnabled(False)
            self.mdetect_btn.setEnabled(False)
            self.mdetect_modify_btn.setEnabled(False)
            self.parameter_detect_lbl.setEnabled(True)
            self.parameter_detect_edt.setEnabled(True)
            self.nuc_segment_btn.setEnabled(True)
            self.nuc_detect_btn.setEnabled(True)
            self.gfilt_water_lbl.setEnabled(True)
            self.gfilt_water_edt.setEnabled(True)
            self.circ_thr_edt.setEnabled(True)
            self.circ_thr_lbl.setEnabled(True)
            self.modify_segm_btn.setEnabled(True)
            self.nuc_track_btn.setEnabled(True)
            self.dist_thr_lbl.setEnabled(True)
            self.dist_thr_edt.setEnabled(True)

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def sld1_update(self):
        """Synchronize all the frames in the main GUI when the slider is moved to browse data"""
        self.time_lbl.setText("time  " + time.strftime("%M:%S", time.gmtime(self.sld1.value() * self.time_step_value)))
        self.frame_numb_lbl.setText("frame  "  +  str(self.sld1.value()))

        if self.data_flag  ==  1:
            self.frame1.setImage(self.filedata.imarray_red[self.sld1.value(), :, :])
            self.frame3.setImage(self.filedata.imarray_green[self.sld1.value(), :, :])
        if self.labbs_flag == 1:
            self.frame2.setImage(np.sign(self.labbs[self.sld1.value(), :, :]))
        if self.nuclei_flag == 1:
            self.frame2.setImage(self.nuclei_labels[self.sld1.value(), :, :], levels=(0, self.nuclei_labels.max()))
        if self.nuclei_t_visual_flag == 1:
            self.frame2.setImage(self.nuclei_t[self.sld1.value(), :, :], levels=(0, self.nuclei_t.max()))
        if self.spots_segm_flag == 1:
            self.frame4.setImage((np.sign(self.spots_3D.spots_ints)[self.sld1.value(), :, :]))

    def parameter_detect_var(self, text):
        """Set the parameter for nuclei detection"""
        self.parameter_detect_value  =  float(text)

    def gfilt2_detect_var(self, text):
        """Set the kernel size for the nuclei pre-smoothing"""
        self.gfilt2_detect_value  =  float(text)

    def gfilt_water_var(self, text):
        """Set the parameter for the watershed algorithm"""
        self.gfilt_water_value  =  int(text)

    def circ_thr_var(self, text):
        """Set the circularity threshold"""
        self.circ_thr_value  =  float(text)

    def dist_thr_var(self, text):
        """Set the distance threshold value for nuclei tracking"""
        self.dist_thr_value  =  int(float(text))

    def spots_thr_var(self, text):
        """Set the threshold value for spots detection"""
        self.spots_thr_value  =  float(text)

    def volume_thr_var(self, text):
        """Set the volume threshold value for the spots discrimination"""
        self.volume_thr_value  =  int(text)

    def time_step_var(self, text):
        """Set the time step value (generally is set automatically by the software)"""
        self.time_step_value  =  float(text)

    def merge_analysis(self):
        """Activate tool to merge partial analysis  results"""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()
        reload(EvolutionController)

        try:
            folder_path  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the folder with the analyzed data"))
            #             flag         =  "Filter"
            flag         =  FlagSpotsDetection.getOldNewFlag()
            print("flag ="  + flag)

            max_dist_ns    =  SpotNcuDistanceThr.getNumb()
            mm             =  MergePartials.MergePartials(folder_path)

            if flag == "Filter":
                print("Filter")
                spots_tracked  =  FakeSpotsRemover.FakeSpotsRemoverMerged(mm.conc_spt, mm.conc_nuc, mm.conc_wild, mm.frames_bm, mm.frames_dm, max_dist_ns).spots_tracked
            else:
                print("No Filter")
                spots_tracked  =  SpotsConnection.SpotsConnection(mm.conc_nuc, mm.conc_spt.astype(int), max_dist_ns).spots_tracked

            evol_check  =  EvolutionController.EvolutionControllerTOT(mm.conc_nuc, mm.conc_wild, spots_tracked, mm.frames_bm, mm.frames_dm, max_dist_ns)
            nuc_active  =  NucleiSpotsConnection.NucleiSpotsConnection4Merged(mm.conc_wild * np.sign(evol_check.conc_nuc_ok), spots_tracked, max_dist_ns)
            self.mpp3   =  MergePartialTool(folder_path, mm, nuc_active, evol_check, spots_tracked, self.nucs_spts_ch, self.soft_version)
            self.mpp3.show()

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def load_merged(self):
        """Load merged analysis"""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            folder_path    =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the folder with the analyzed data"))
            mm             =  LoadMergedResults.LoadMergedResults_mm(folder_path)
            nuc_active     =  LoadMergedResults.LoadMergedResultsNucActive(folder_path)
            evol_check     =  LoadMergedResults.LoadMergedResultsEvolCheck(folder_path)
            spots_tracked  =  SpotsFeaturesLoader.SpotsFeaturesLoader(folder_path + '/spots_tracked_coords.npy').spts_mtx
            self.mpp6      =  MergePartialTool(folder_path, mm, nuc_active, evol_check, spots_tracked, self.nucs_spts_ch, self.soft_version)
            self.mpp6.show()

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def popup_nuclei_raw(self):
        """Popup tool to visualize raw data nuclei"""
        PopUpTool.PopUpTool([self.filedata.imarray_red, 'Nuclei Raw Data'])

    def popup_nuclei_detected(self):
        """Popup tool to visualize detected nuclei"""
        PopUpTool.PopUpTool([np.sign(self.labbs), 'Detected Nuclei'])

    def popup_nuclei_segmented(self):
        """Popup tool to visualize segmented nuclei"""
        PopUpTool.PopUpTool([self.nuclei_labels, 'Segmented Nuclei', self.mycmap])

    def popup_nuclei_trackeded(self):
        """Popup tool to visualize raw data nuclei"""
        PopUpTool.PopUpTool([self.nuclei_t, 'Segmented Nuclei', self.mycmap])

    def popup_spots_raw(self):
        """Popup tool to visualize raw data spots"""
        PopUpTool.PopUpTool([self.filedata.imarray_green, 'Spots Raw Data'])

    def popup_spots_segm(self):
        """Popup tool to visualize segmented spots"""
        PopUpTool.PopUpTool([self.spots_segm, 'Segmented Spots'])

    def popup_nucactive(self):
        """Popup tool to visualize active nuclei"""
        pg.image(self.nuclei_active_visual, title="Active Nuclei")
        pg.plot(self.n_active_vector, pen='r', symbol='x')

    def nuclei_detection(self):
        """Nuclei detection"""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            #             self.labbs  =  NucleiDetect.NucleiDetect(self.imarray_red, self.parameter_detect_value).labbs

            if self.gaus_log_detect_value == "Log Flt":
                self.labbs  =  NucleiDetectLog.NucleiDetectLog(self.filedata.imarray_red, self.parameter_detect_value).labbs
            else:
                self.labbs  =  NucleiDetect.NucleiDetect(self.filedata.imarray_red, self.parameter_detect_value).labbs

            self.frame2.setImage(np.sign(self.labbs[self.sld1.value(), :, :]))
            self.labbs_flag            =  1
            self.nuclei_flag           =  0
            self.nuclei_t_visual_flag  =  0

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def nuclei_segmentation(self):
        """Segment nuclei"""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            self.nuclei_labels  =  NucleiSegmentStackMultiCore.NucleiSegmentStackMultiCore(self.labbs, self.circ_thr_value, self.gfilt_water_value).nuclei_labels
            self.frame2.setImage(self.nuclei_labels[self.sld1.value(), :, :], levels=(0, self.nuclei_labels.max()))
            self.mycmap  =  pg.ColorMap(np.linspace(0, 1, self.nuclei_labels.max()), color=self.colors4map)
            self.frame2.setColorMap(self.mycmap)
            self.labbs_flag            =  0
            self.nuclei_flag           =  1
            self.nuclei_t_visual_flag  =  0

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def nuclei_tracking(self):
        """Track nuclei"""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            reload(NucleiConnectMultiCore)
            self.nuclei_t  =  NucleiConnectMultiCore.NucleiConnectMultiCore(self.nuclei_labels, self.dist_thr_value).nuclei_tracked
            self.frame2.setImage(self.nuclei_t[self.sld1.value(), :, :], levels=(0, self.nuclei_t.max()))
            self.mycmap  =  pg.ColorMap(np.linspace(0, 1, self.nuclei_t.max()), color=self.colors4map)
            self.frame2.setColorMap(self.mycmap)
            self.labbs_flag            =  0
            self.nuclei_flag           =  0
            self.nuclei_t_visual_flag  =  1

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def spots_detect(self):
        """Detect spots"""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()
        reload(SpotsDetectionChopper)

        try:
            self.spots_3D     =  SpotsDetectionChopper.SpotsDetectionChopper(self.filedata.green4D.astype(float), self.spots_thr_value, self.volume_thr_value)
            self.spots_segm_flag    =  1
            self.frame4.setImage(np.sign(self.spots_3D.spots_ints)[self.sld1.value(), :, :])
            self.nuc_spots_conn()

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def nuc_spots_conn(self):
        """Connect tracked nuclei and detected spots"""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            self.max_dist_ns           =  SpotNcuDistanceThr.getNumb()
            self.spots_tracked         =  SpotsConnection.SpotsConnection(self.nuclei_t, np.sign(self.spots_3D.spots_ints), self.max_dist_ns).spots_tracked
            nuc_active                 =  NucleiSpotsConnection.NucleiSpotsConnection(self.spots_tracked, self.nuclei_t)
            self.nuclei_active         =  nuc_active.nuclei_active
            self.nuclei_active_visual  =  nuc_active.nuclei_active3c
            self.n_active_vector       =  nuc_active.n_active_vector
            pg.image(self.nuclei_active_visual)

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def modify_cycle_tool(self):
        """Activate the tool for manual corrections to the before or after mitosis part"""
        self.mpp  =  ModifierCycleTool(self.filedata.imarray_red, np.copy(self.nuclei_labels), self.sld1.value())
        self.mpp.show()
        self.mpp.procStart.connect(self.sgnl_update_cycle)

    def sgnl_update_cycle(self, message):
        """Update manual corrections from the popup tool to the main GUI"""
        self.nuclei_labels  =  self.mpp.nuclei_labels
        self.frame2.setImage(self.nuclei_labels)

        self.mpp.close()

        self.time_lbl.setText("time  " + time.strftime("%M:%S", time.gmtime(self.sld1.value() * self.time_step_value)))
        self.frame_numb_lbl.setText("frame  "  +  str(self.sld1.value()))

        self.sld1.setSliderPosition(message)

        self.frame1.setImage(self.filedata.imarray_red[message, :, :])
        self.frame3.setImage(self.filedata.imarray_green[message, :, :])
        self.frame2.setImage(self.nuclei_labels[message, :, :])
        if self.spots_segm_flag == 1:
            self.frame4.setImage(self.spots_segm[message, :, :])

    def modify_mitosis_tool(self):
        """Activate the tool to track nuclei during mitosis"""
        self.mpp2  =  ModifierMitosisTool(self.filedata.imarray_red, self.labbs, np.zeros(self.labbs.shape, dtype=np.uint16), self.sld1.value())
        self.mpp2.show()
        self.mpp2.procStart.connect(self.sgnl_update_mitosis)

    def sgnl_update_mitosis(self, message):
        """"Update manual corrections from the popup tool to the main GUI"""
        self.nuclei_t       =  self.mpp2.mtx_tot.astype(np.int16)
        self.nuclei_labels  =  self.mpp2.mtx_tot.astype(np.int16)
        self.labbs          =  self.mpp2.labbs.astype(np.uint16)
        self.mpp2.close()
        self.labbs_flag            =  0
        self.nuclei_t_visual_flag  =  1

        self.time_lbl.setText("time  " + time.strftime("%M:%S", time.gmtime(self.sld1.value() * self.time_step_value)))
        self.frame_numb_lbl.setText("frame  "  +  str(self.sld1.value()))

        self.sld1.setSliderPosition(message)

        self.frame1.setImage(self.filedata.imarray_red[message, :, :])
        self.frame3.setImage(self.filedata.imarray_green[message, :, :])
        self.frame2.setImage(self.nuclei_t[message, :, :])
        self.mycmap  =  pg.ColorMap(np.linspace(0, 1, self.nuclei_labels.max()), color=self.colors4map)
        self.frame2.setColorMap(self.mycmap)

        if self.spots_segm_flag == 1:
            self.frame4.setImage(self.spots_segm[message, :, :])

    def spots_visual(self):
        """Activate tool to visualize the performed spots detection"""
        self.spsp  =  SpotsAnalyser(self.filedata.imarray_green, self.spots_tracked)
        self.spsp.show()

    def mdetect(self):
        """Detect nuclei during mitosis"""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            labbs  =  NucleiDetect.NucleiDetect(self.filedata.imarray_red, self.gfilt2_detect_value).labbs
            steps  =  labbs.shape[0]

            for t in range(steps):
                labbs[t, :, :]  =  skmr.remove_small_objects(labbs[t, :, :].astype(int), 50)

            self.labbs  =  labbs
            self.frame2.setImage(np.sign(self.labbs[self.sld1.value(), :, :]))
            self.labbs_flag            =  1
            self.nuclei_flag           =  0
            self.nuclei_t_visual_flag  =  0

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def save_partial(self):
        """Save partial analysis"""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:

            if self.bm_dm_am == 1:
                flagname  =  "b"
                file  =  open(self.folder_storedata + '/before_mitosis_params.txt', "w")
                if self.gaus_log_detect_value == "Gauss Flt":
                    file.write("Gaussian Filter Kernel Size = " + str(self.parameter_detect_value))
                else:
                    file.write("Log Detection Threshold Coefficient = " + str(self.parameter_detect_value))

                file.write('\n' + "Watershed Parameter = " + str(self.gfilt_water_value))
                file.write('\n' + "Circularity Threshold = " + str(self.circ_thr_value))
                file.write('\n' + "Distance Threshold = " + str(self.dist_thr_value))
                file.write('\n' + "Spots Threshold = " + str(self.spots_thr_value))
                file.write('\n' + "Volume Threshold = " + str(self.volume_thr_value))
                file.write('\n' + "Time step = " + str(self.time_step_value))
                file.write('\n' + "Spots removed up to frame = " + str(self.t_track_end_value))
                file.close()

            elif self.bm_dm_am == 2:
                flagname  =  "d"
                file  =  open(self.folder_storedata + '/during_mitosis_params.txt', "w")
                file.write("Gaussian Filter Kernel Size = " + str(self.gfilt2_detect_value))
                file.write('\n' + "Spots Threshold = " + str(self.spots_thr_value))
                file.write('\n' + "Size Threshold = " + str(self.mpp2.size_thr_value))
                file.write('\n' + "Overlap ratio = " + str(self.mpp2.overlap_ratio_value))
                file.write('\n' + "Proliferation ratio = " + str(self.mpp2.prolif_ratio_value))
                file.write('\n' + "Distance Threshold = " + str(self.mpp2.dist_singtrck_value))
                file.write('\n' + "Spots Threshold = " + str(self.spots_thr_value))
                file.write('\n' + "Volume Threshold = " + str(self.volume_thr_value))
                file.write('\n' + "Time step = " + str(self.time_step_value))
                file.write('\n' + "Spots removed up to frame = " + str(self.t_track_end_value))
                file.close()
                np.savez_compressed(self.folder_storedata + "/lbl_segm_" + flagname + "m.npz", self.labbs)

            elif self.bm_dm_am == 3:
                flagname  =  "a"
                file  =  open(self.folder_storedata + '/after_mitosis_params.txt', "w")
                if self.gaus_log_detect_value == "Gauss Flt":
                    file.write("Gaussian Filter Kernel Size = " + str(self.parameter_detect_value))
                else:
                    file.write("Log Detection Threshold Coefficient = " + str(self.parameter_detect_value))

                file.write('\n' + "Watershed Parameter = " + str(self.gfilt_water_value))
                file.write('\n' + "Circularity Threshold = " + str(self.circ_thr_value))
                file.write('\n' + "Distance Threshold = " + str(self.dist_thr_value))
                file.write('\n' + "Spots Threshold = " + str(self.spots_thr_value))
                file.write('\n' + "Volume Threshold = " + str(self.volume_thr_value))
                file.write('\n' + "Time step = " + str(self.time_step_value))
                file.write('\n' + "Spots removed up to frame = " + str(self.t_track_end_value))
                file.close()

            np.savez_compressed(self.folder_storedata + "/lbl_nuclei_" + flagname + "m.npz", self.nuclei_t)
            SpotsFeaturesSaver.SpotsFeaturesSaver(self.spots_tracked, self.folder_storedata + "/trk_spots_" + flagname + "m_coords.npy")
            SpotsFeaturesSaver.SpotsFeaturesSaver(self.spots_3D.spots_ints, self.folder_storedata + "/ints_spots_" + flagname + "m_coords.npy")
            SpotsFeaturesSaver.SpotsFeaturesSaver(self.spots_3D.spots_vol, self.folder_storedata + "/vol_spots_" + flagname + "m_coords.npy")
            np.save(self.folder_storedata + "/tzxy_spots_" + flagname + "m.npy", self.spots_3D.spots_tzxy)
            np.save(self.folder_storedata + "/time_step_value.npy", np.array([self.time_step_value]))

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def tile_coordinates(self):
        tile_info   =  TileMap.getNumb()
        tile_fname  =  str(QtWidgets.QFileDialog.getOpenFileName(None, "Select lsm tile data files (maximum intensity projected)", filter="*.tif *.lsm *.czi"))
        txt_path    =  QtWidgets.QFileDialog.getOpenFileName(None, "Define a .txt file in which write")
        FromTile2GlobCoordinate.FromTile2GlobCoordinate(tile_fname, tile_info, txt_path)

    def set_color_channel(self):
        """Call pop up tool to set channels"""
        self.nucs_spts_ch  =  SetColorChannel.getChannels() - 1

    def rmv_mitoticalTS(self):
        """Call PopUpTool up tool to remove mitotical spots"""
        if hasattr(self.spots_3D, 'spots_coords') == True:
            self.mpp8  =  RemoveMitoticalSpots(self.filedata, self.spots_3D)
            self.mpp8.show()
            self.mpp8.procStart.connect(self.update_mip_spots_sgnl)
        else:
            self.msg = MessageSpotsCoords()
            self.msg.show()

    def update_mip_spots_sgnl(self):
        """Update the work of RemoveMitoticalSpots in the main GUI"""
        self.spots_3D                =  self.mpp8.spots_3D
        self.t_track_end_value       =  self.mpp8.t_track_end_value
        self.spots_segm_flag         =  1
        self.frame4.setImage(np.sign(self.spots_3D.spots_ints[self.sld1.value(), :, :]))
        self.frame3.updateImage()
        self.mpp8.close()
        self.nuc_spots_conn()

    def false_clc_makeup(self):
        """Write multi tiff file of the false colored movie"""
        folder_name  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the folder with the analyzed data"))
        FalseColoredMakeUp.FalseColoredMakeUp(folder_name)

    def check_results(self):
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            folder_name  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the folder with the analyzed data"))
            self.mpp9    =  CheckResults(folder_name)
            self.mpp9.show()
        except Exception:
            traceback.print_exc()

        self.ready_indicator()


class ChopStack(QtWidgets.QWidget):
    """Popup tool to chop the stack and set first and last mitosis frame"""
    def __init__(self, imarray_red, imarray_green, green4D, time_step_value, nucs_spts_ch, folder_storedata):
        QtWidgets.QWidget.__init__(self)

        self.imarray_red       =  imarray_red
        self.imarray_green     =  imarray_green
        self.green4D           =  green4D
        self.folder_storedata  =  folder_storedata
        self.time_step_value   =  time_step_value
        self.nucs_spts_ch      =  nucs_spts_ch

        im2show  =  np.zeros(self.imarray_green.shape + (3,))
        im2show[:, :, :, 0]  =  self.imarray_red / self.imarray_red.max()
        im2show[:, :, :, 1]  =  2 * self.imarray_green / self.imarray_green.max()

        framepp1  =  pg.ImageView(self, name='Frame1')
        framepp1.ui.roiBtn.hide()
        framepp1.ui.menuBtn.hide()
        framepp1.ui.histogram.hide()
        framepp1.setImage(im2show)
        framepp1.timeLine.sigPositionChanged.connect(self.timelabel)

        mtss_st_btn  =  QtWidgets.QPushButton("Mitosis Start", self)
        mtss_st_btn.clicked.connect(self.mtss_st)
        mtss_st_btn.setToolTip('Frame in which mitosis starts ')
        mtss_st_btn.setFixedSize(110, 25)

        mtss_st_lbl  =  QtWidgets.QLabel(self)
        mtss_st_lbl.setFixedSize(110, 25)
        mtss_st_lbl.setText(" Start ")

        mtss_end_btn  =  QtWidgets.QPushButton("Mitosis End", self)
        mtss_end_btn.clicked.connect(self.mtss_end)
        mtss_end_btn.setToolTip('Frame in which mitosis ends')
        mtss_end_btn.setFixedSize(110, 25)

        mtss_end_lbl  =  QtWidgets.QLabel(self)
        mtss_end_lbl.setFixedSize(110, 25)
        mtss_end_lbl.setText(" End ")

        done_btn  =  QtWidgets.QPushButton("Done", self)
        done_btn.clicked.connect(self.done)
        done_btn.setToolTip('Save the selection')
        done_btn.setFixedSize(110, 25)

        time_lbl  =  QtWidgets.QLabel("time  " + '0', self)
        time_lbl.setFixedSize(110, 25)

        frame_numb_lbl  =  QtWidgets.QLabel("frame  " + '0', self)
        frame_numb_lbl.setFixedSize(110, 25)

        tools  =  QtWidgets.QVBoxLayout()
        tools.addWidget(mtss_st_btn)
        tools.addWidget(mtss_st_lbl)
        tools.addStretch()
        tools.addWidget(mtss_end_btn)
        tools.addWidget(mtss_end_lbl)
        tools.addStretch()
        tools.addWidget(done_btn)
        tools.addStretch()
        tools.addWidget(time_lbl)
        tools.addWidget(frame_numb_lbl)

        layout  =  QtWidgets.QHBoxLayout()
        layout.addWidget(framepp1)
        layout.addLayout(tools)

        self.setLayout(layout)
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle("Chop Tool")

        self.framepp1        =  framepp1
        self.mtss_st_lbl     =  mtss_st_lbl
        self.mtss_end_lbl    =  mtss_end_lbl
        self.time_lbl        =  time_lbl
        self.frame_numb_lbl  =  frame_numb_lbl

    def mtss_st(self):
        """Set first mitosis frames"""
        mtss_st_val       =  self.framepp1.currentIndex
        self.mtss_st_val  =  mtss_st_val
        self.mtss_st_lbl.setText("  Start  " + str(self.mtss_st_val))

    def mtss_end(self):
        """Set the last mitosis frame"""
        mtss_end_val       =  self.framepp1.currentIndex
        self.mtss_end_val  =  mtss_end_val
        self.mtss_end_lbl.setText("  End  " + str(self.mtss_end_val))

    def done(self):
        """Close the popup tool and write the data in the analysis folder"""
        np.save(self.folder_storedata + '/imarray_red_whole.npy', self.imarray_red)
        np.save(self.folder_storedata + '/green4D_whole.npy', self.green4D)
        np.save(self.folder_storedata + '/mtss_boundaries.npy', np.array([self.mtss_st_val, self.mtss_end_val]))
        np.save(self.folder_storedata + "/nucs_s_ch.npy", self.nucs_spts_ch)
        np.save(self.folder_storedata + "/time_step_value.npy", np.array([self.time_step_value]))

        self.close()

    def timelabel(self):
        """Time label updater"""
        self.time_lbl.setText("time  " + time.strftime("%M:%S", time.gmtime(self.framepp1.currentIndex * self.time_step_value)))
        self.frame_numb_lbl.setText("frame  "  +  str(self.framepp1.currentIndex))


class ModifierCycleTool(QtWidgets.QWidget):
    """Popup tool to manually modify before or after mitosis nuclei segmentation"""
    procStart = QtCore.pyqtSignal(int)

    def __init__(self, imarray_red, nuclei_labels, cif_start):
        QtWidgets.QWidget.__init__(self)

        self.imarray_red    =  imarray_red
        self.nuclei_labels  =  nuclei_labels
        self.cif_start      =  cif_start

        frameShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.ShiftModifier + QtCore.Qt.Key_End), self)
        frameShortcut.activated.connect(self.shuffle_clrs)

        tabs  =  QtWidgets.QTabWidget()
        tab1  =  QtWidgets.QWidget()
        tab2  =  QtWidgets.QWidget()

        mycmap         =  np.fromfile("mycmap.bin", "uint16").reshape((10000, 3))
        self.colors4map  =  []
        for k in range(mycmap.shape[0]):
            self.colors4map.append(mycmap[k, :])
        self.colors4map[0]  =  np.array([0, 0, 0])

        framepp1  =  pg.ImageView(self, name='Frame1')
        framepp1.getImageItem().mouseClickEvent  =  self.click
        framepp1.ui.roiBtn.hide()
        framepp1.ui.menuBtn.hide()
        framepp1.setImage(self.nuclei_labels)
        mycmap  =  pg.ColorMap(np.linspace(0, 1, self.nuclei_labels.max()), color=self.colors4map)
        framepp1.setColorMap(mycmap)
        framepp1.timeLine.sigPositionChanged.connect(self.update_frame2)

        framepp2  =  pg.ImageView(self)
        framepp2.ui.roiBtn.hide()
        framepp2.ui.menuBtn.hide()
        framepp2.setImage(self.imarray_red)
        framepp2.timeLine.sigPositionChanged.connect(self.update_frame1)
        framepp2.view.setXLink('Frame1')
        framepp2.view.setYLink('Frame1')

        modify_btn  =  QtWidgets.QPushButton("Modify", self)
        modify_btn.setFixedSize(120, 25)
        modify_btn.clicked.connect(self.modify_lbls)

        update_mainwindows_btn  =  QtWidgets.QPushButton("Update Nuclei", self)
        update_mainwindows_btn.setFixedSize(120, 25)
        update_mainwindows_btn.clicked.connect(self.update_mainwindows)

        shuffle_clrs_btn  =  QtWidgets.QPushButton("Shuffle Colors", self)
        shuffle_clrs_btn.setFixedSize(120, 25)
        shuffle_clrs_btn.clicked.connect(self.shuffle_clrs)
        shuffle_clrs_btn.setToolTip('Shuffle colors')

        frame_numb_lbl  =  QtWidgets.QLabel("frame  " + '0', self)
        frame_numb_lbl.setFixedSize(110, 25)

        frame1_box  =  QtWidgets.QHBoxLayout()
        frame1_box.addWidget(framepp1)

        frame2_box  =  QtWidgets.QHBoxLayout()
        frame2_box.addWidget(framepp2)

        btn_box  =  QtWidgets.QHBoxLayout()
        btn_box.addWidget(shuffle_clrs_btn)
        btn_box.addStretch()
        btn_box.addWidget(frame_numb_lbl)
        btn_box.addWidget(modify_btn)
        btn_box.addWidget(update_mainwindows_btn)

        tab1.setLayout(frame1_box)
        tab2.setLayout(frame2_box)

        tabs.addTab(tab1, "Segmented")
        tabs.addTab(tab2, "Raw")

        layout  =  QtWidgets.QVBoxLayout()
        layout.addWidget(tabs)                                                                                          # tabs is a Widget not a Layout!!!!!
        layout.addLayout(btn_box)

        self.framepp1        =  framepp1
        self.framepp2        =  framepp2
        self.mycmap          =  mycmap
        self.c_count         =  0
        self.frame_numb_lbl  =  frame_numb_lbl

        self.setLayout(layout)
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle("Modify Segmentation Tool")

        self.framepp1.setCurrentIndex(self.cif_start)

    def keyPressEvent(self, event):
        """Control-Z for nuclei manual correction"""
        if event.key() == (Qt.ControlModifier and Qt.Key_Z):
            # cif                            =  self.framepp1.currentIndex
            # hh                             =  self.framepp1.view.viewRange()
            # self.nuclei_labels[cif, :, :]  =  self.bufframe
            # self.framepp1.setImage(self.nuclei_labels)
            # self.framepp1.setCurrentIndex(cif)
            # self.framepp1.view.setXRange(hh[0][0], hh[0][1], padding=.0002)
            # self.framepp1.view.setYRange(hh[1][0], hh[1][1], padding=.0002)

            cif                            =  self.framepp1.currentIndex
            self.nuclei_labels[cif, :, :]  =  self.bufframe
            self.framepp1.updateImage()

        if event.key() == (QtCore.Qt.ControlModifier and Qt.Key_Delete):
            self.modify_lbls()

    def click(self, event):
        """Defining mouse interaction: positioning the segment on the frame"""
        event.accept()
        pos        =  event.pos()
        modifiers  =  QtWidgets.QApplication.keyboardModifiers()

        if modifiers  ==  QtCore.Qt.ShiftModifier:
            if self.c_count - 2 * (self.c_count // 2) == 0:
                self.pos1          =   pos
            else:
                try:
                    self.framepp1.removeItem(self.roi)
                except AttributeError:
                    pass

                self.roi      =  pg.LineSegmentROI([self.pos1, pos], pen='r')
                self.framepp1.addItem(self.roi)

            self.c_count += 1

    def modify_lbls(self):
        """Modify the segmentattion"""
        # cif      =  self.framepp1.currentIndex
        # hh       =  self.framepp1.view.viewRange()
        # pp       =  self.roi.getHandles()
        # pp       =  [self.roi.mapToItem(self.framepp1.imageItem, p.pos()) for p in pp]
        # end_pts  =  np.array([[int(pp[0].x()), int(pp[0].y())], [int(pp[1].x()), int(pp[1].y())]])

        # bufframe                       =  np.copy(self.nuclei_labels[cif, :, :])
        # self.nuclei_labels[cif, :, :]  =  LabelsModify.LabelsModify(self.nuclei_labels[cif, :, :], end_pts).labels_fin
        # self.framepp1.setImage(self.nuclei_labels)
        # self.framepp1.setCurrentIndex(cif)
        # self.framepp1.view.setXRange(hh[0][0], hh[0][1], padding=.0002)
        # self.framepp1.view.setYRange(hh[1][0], hh[1][1], padding=.0002)
        # self.bufframe  =  bufframe
        # self.framepp1.removeItem(self.roi)

        cif      =  self.framepp1.currentIndex
        pp       =  self.roi.getHandles()
        pp       =  [self.roi.mapToItem(self.framepp1.imageItem, p.pos()) for p in pp]
        end_pts  =  np.array([[int(pp[0].x()), int(pp[0].y())], [int(pp[1].x()), int(pp[1].y())]])

        bufframe                       =  np.copy(self.nuclei_labels[cif, :, :])
        self.nuclei_labels[cif, :, :]  =  LabelsModify.LabelsModify(self.nuclei_labels[cif, :, :], end_pts).labels_fin
        # self.framepp1.removeItem(self.roi)
        self.framepp1.updateImage()
        self.bufframe  =  bufframe

    def update_frame1(self):
        """Synchronize frame 1 index with the 2"""
        self.framepp1.setCurrentIndex(self.framepp2.currentIndex)
        self.frame_numb_lbl.setText("frame  "  +  str(self.framepp1.currentIndex))

    def update_frame2(self):
        """Synchronize frame 2 index with the 1"""
        self.framepp2.setCurrentIndex(self.framepp1.currentIndex)
        self.frame_numb_lbl.setText("frame  "  +  str(self.framepp1.currentIndex))

    def shuffle_clrs(self):
        """Shuffle colors"""
        colors_bff  =  self.colors4map[1:]
        np.random.shuffle(colors_bff)
        self.colors4map[1:]  =  colors_bff
        mycmap  =  pg.ColorMap(np.linspace(0, 1, self.nuclei_labels.max()), color=self.colors4map)
        self.framepp1.setColorMap(mycmap)

    @QtCore.pyqtSlot()
    def update_mainwindows(self):
        """Send signal to the main GUI for updating"""
        val  =  self.framepp1.currentIndex
        self.procStart.emit(val)


class ModifierMitosisTool(QtWidgets.QWidget):
    """Popup tool to manually modify the tracking during mitosis."""
    procStart  =  QtCore.pyqtSignal(int)

    def __init__(self, imarray_red, labbs, mtx_tot, cif_start):
        QtWidgets.QWidget.__init__(self)

        self.labbs        =  labbs
        self.labbs_start  =  np.copy(self.labbs)
        self.imarray_red  =  imarray_red
        self.cif_start    =  cif_start

        frameShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.ShiftModifier + QtCore.Qt.Key_End), self)
        frameShortcut.activated.connect(self.shuffle_clrs)

        tabs  =  QtWidgets.QTabWidget()
        tab1  =  QtWidgets.QWidget()
        tab2  =  QtWidgets.QWidget()

        framepp1  =  pg.ImageView(self, name='Frame1')
        framepp1.ui.roiBtn.hide()
        framepp1.ui.menuBtn.hide()
#         framepp1.ui.histogram.hide()
#         framepp1.setImage(self.labbs_start)
        framepp1.getImageItem().mouseClickEvent  =  self.click
        framepp1.timeLine.sigPositionChanged.connect(self.update_frame2)

        framepp2  =  pg.ImageView(self)
        framepp2.ui.roiBtn.hide()
        framepp2.ui.menuBtn.hide()
        framepp2.setImage(self.imarray_red)
        framepp2.timeLine.sigPositionChanged.connect(self.update_frame1)
        framepp2.view.setXLink('Frame1')
        framepp2.view.setYLink('Frame1')

        automatic_segm_btn  =  QtWidgets.QPushButton("Segm Frames", self)
        automatic_segm_btn.clicked.connect(self.auto_segm)
        automatic_segm_btn.setToolTip('Automatic segmentation of nuclei')
        automatic_segm_btn.setFixedSize(110, 25)

        circ_thr_lbl  =  QtWidgets.QLabel('Circ Thr', self)
        circ_thr_lbl.setFixedSize(60, 25)

        circ_thr_edt  =  QtWidgets.QLineEdit(self)
        circ_thr_edt.textChanged[str].connect(self.circ_thr_var)
        circ_thr_edt.setToolTip('Circularity Threshold of the detected nuclei (suggested value is 0.65)')
        circ_thr_edt.setFixedSize(35, 25)

        gfilt_water_lbl  =  QtWidgets.QLabel('W-Shed', self)
        gfilt_water_lbl.setFixedSize(60, 25)

        gfilt_water_edt  =  QtWidgets.QLineEdit(self)
        gfilt_water_edt.textChanged[str].connect(self.gfilt_water_var)
        gfilt_water_edt.setToolTip('Set the size parameter for the Water Shed algorithm (suggested value 7)')
        gfilt_water_edt.setFixedSize(35, 25)

        hor_line_one  =  QtWidgets.QFrame()
        hor_line_one.setFrameStyle(QtWidgets.QFrame.HLine)

        mmodify_segm_btn  =  QtWidgets.QPushButton("Modify", self)
        mmodify_segm_btn.clicked.connect(self.mmodify_segm)
        mmodify_segm_btn.setToolTip('Cuts the Speckles Following the ROI')
        mmodify_segm_btn.setFixedSize(110, 25)

        mdone_btn  =  QtWidgets.QPushButton("Done", self)
        mdone_btn.clicked.connect(self.mdone)
        mdone_btn.setToolTip('Automatic Tracking')
        mdone_btn.setFixedSize(110, 25)

        cut_left_tggl  =  QtWidgets.QCheckBox('Cut Left', self)
        cut_left_tggl.stateChanged.connect(self.cut_left)
        cut_left_tggl.setFixedSize(110, 25)
        cut_left_tggl.setToolTip("Ctrl+3")

        rmv_wr_tggl  =  QtWidgets.QCheckBox('Rmv Wrong', self)
        rmv_wr_tggl.stateChanged.connect(self.remove_wr)
        rmv_wr_tggl.setFixedSize(110, 25)
        rmv_wr_tggl.setToolTip("Ctrl+2")

        pickcolor_tggl  =  QtWidgets.QCheckBox('Pick Label', self)
        pickcolor_tggl.stateChanged.connect(self.pickcolor)
        pickcolor_tggl.setFixedSize(110, 25)
        pickcolor_tggl.setToolTip("Ctrl+1")

        givecolor_tggl  =  QtWidgets.QCheckBox('Give Label', self)
        givecolor_tggl.stateChanged.connect(self.givecolor)
        givecolor_tggl.setFixedSize(110, 25)
        givecolor_tggl.setToolTip("Ctrl+1")

        dist_singtrck_edt  =  QtWidgets.QLineEdit(self)
        dist_singtrck_edt.setToolTip("Distance Threshold to follow the Nucleus")
        dist_singtrck_edt.setFixedSize(35, 22)
        dist_singtrck_edt.textChanged[str].connect(self.dist_singtrck_var)
        dist_singtrck_edt.setText("20")

        dist_singtrck_lbl  =  QtWidgets.QLabel("Dist Thr", self)
        dist_singtrck_lbl.setFixedSize(70, 25)

        overlap_ratio_edt  =  QtWidgets.QLineEdit(self)
        overlap_ratio_edt.setToolTip("Overlapping threshold: if it is smaller, it happen by chance (suggested value 0.2)")
        overlap_ratio_edt.setFixedSize(35, 22)
        overlap_ratio_edt.textChanged[str].connect(self.overlap_ratio_var)
        overlap_ratio_edt.setText("0.2")

        overlap_ratio_lbl  =  QtWidgets.QLabel("Overlap", self)
        overlap_ratio_lbl.setFixedSize(70, 25)

        wshed_box  =  QtWidgets.QHBoxLayout()
        wshed_box.addWidget(gfilt_water_lbl)
        wshed_box.addWidget(gfilt_water_edt)

        circ_thr_box  =  QtWidgets.QHBoxLayout()
        circ_thr_box.addWidget(circ_thr_lbl)
        circ_thr_box.addWidget(circ_thr_edt)

        overlap_box  =  QtWidgets.QHBoxLayout()
        overlap_box.addWidget(overlap_ratio_edt)
        overlap_box.addWidget(overlap_ratio_lbl)

        prolif_ratio_edt  =  QtWidgets.QLineEdit(self)
        prolif_ratio_edt.setToolTip("Proliferation threshold: if it is higher, the overlap is due to proliferation (suggested value 0.4)")
        prolif_ratio_edt.setFixedSize(35, 22)
        prolif_ratio_edt.textChanged[str].connect(self.prolif_ratio_var)
        prolif_ratio_edt.setText("0.4")

        prolif_ratio_lbl  =  QtWidgets.QLabel("Prolif", self)
        prolif_ratio_lbl.setFixedSize(70, 25)

        prolif_box  =  QtWidgets.QHBoxLayout()
        prolif_box.addWidget(prolif_ratio_edt)
        prolif_box.addWidget(prolif_ratio_lbl)

        size_thr_edt  =  QtWidgets.QLineEdit(self)
        size_thr_edt.setToolTip("Size Threshold: objects with a surface smaller than this will be discarded")
        size_thr_edt.setFixedSize(35, 22)
        size_thr_edt.textChanged[str].connect(self.size_thr_var)
        size_thr_edt.setText("60")

        size_thr_lbl  =  QtWidgets.QLabel("Size Thr", self)
        size_thr_lbl.setFixedSize(70, 25)

        size_thr_box  =  QtWidgets.QHBoxLayout()
        size_thr_box.addWidget(size_thr_edt)
        size_thr_box.addWidget(size_thr_lbl)

        write_data_btn  =  QtWidgets.QPushButton("Update Mitosis", self)
        write_data_btn.clicked.connect(self.update_mainwindows2)
        write_data_btn.setToolTip('Save the Result Matrix Data')
        write_data_btn.setFixedSize(110, 25)

        checkwork_btn  =  QtWidgets.QPushButton("Check Tracking", self)
        checkwork_btn.clicked.connect(self.checkwork)
        checkwork_btn.setToolTip('Check if your tracking is correctly done')
        checkwork_btn.setFixedSize(110, 25)

        shuffle_clrs_btn  =  QtWidgets.QPushButton("Shuffle Colors", self)
        shuffle_clrs_btn.clicked.connect(self.shuffle_clrs)
        shuffle_clrs_btn.setToolTip('Shuffle colors')
        shuffle_clrs_btn.setFixedSize(110, 25)

        frame_numb_lbl  =  QtWidgets.QLabel("frame  " + '0', self)
        frame_numb_lbl.setFixedSize(110, 25)

        frame1_box  =  QtWidgets.QHBoxLayout()
        frame1_box.addWidget(framepp1)

        frame2_box  =  QtWidgets.QHBoxLayout()
        frame2_box.addWidget(framepp2)

        tab1.setLayout(frame1_box)
        tab2.setLayout(frame2_box)

        tabs.addTab(tab1, "Segmented")
        tabs.addTab(tab2, "Raw")

        dist_singtrck_box  =  QtWidgets.QHBoxLayout()
        dist_singtrck_box.addWidget(dist_singtrck_edt)
        dist_singtrck_box.addWidget(dist_singtrck_lbl)

        tggls  =  QtWidgets.QVBoxLayout()
        tggls.addWidget(cut_left_tggl)
        tggls.addWidget(rmv_wr_tggl)
        tggls.addWidget(pickcolor_tggl)
        tggls.addWidget(givecolor_tggl)

        commands  =  QtWidgets.QVBoxLayout()
        commands.addLayout(wshed_box)
        commands.addLayout(circ_thr_box)
        commands.addWidget(automatic_segm_btn)
        commands.addWidget(hor_line_one)
        commands.addWidget(mmodify_segm_btn)
        commands.addLayout(size_thr_box)
        commands.addLayout(overlap_box)
        commands.addLayout(prolif_box)
        commands.addLayout(dist_singtrck_box)
        commands.addWidget(mdone_btn)
        commands.addStretch()
        commands.addLayout(tggls)
        commands.addStretch()
        commands.addWidget(shuffle_clrs_btn)
        commands.addWidget(checkwork_btn)
        commands.addStretch()
        commands.addWidget(write_data_btn)
        commands.addStretch()
        commands.addWidget(frame_numb_lbl)

        layout  =  QtWidgets.QHBoxLayout()
        layout.addWidget(tabs)
        layout.addLayout(commands)

        mycmap         =  np.fromfile("mycmap.bin", "uint16").reshape((10000, 3))
        self.colors4map  =  []
        for k in range(mycmap.shape[0]):
            self.colors4map.append(mycmap[k, :])
        self.colors4map[0]  =  np.array([0, 0, 0])
        self.colors4map[1]  =  np.array([255, 0, 0])
        self.colors4map[2]  =  np.array([0, 0, 255])

        self.framepp1  =  framepp1
        self.framepp2  =  framepp2

        self.pickcolor_tggl     =  pickcolor_tggl
        self.rmv_wr_tggl        =  rmv_wr_tggl
        self.givecolor_tggl     =  givecolor_tggl
        self.cut_left_tggl      =  cut_left_tggl
        self.dist_singtrck_lbl  =  dist_singtrck_lbl
        self.dist_singtrck_edt  =  dist_singtrck_edt
        self.mdone_btn          =  mdone_btn
        self.frame_numb_lbl     =  frame_numb_lbl
        self.overlap_ratio_edt  =  overlap_ratio_edt
        self.prolif_ratio_edt   =  prolif_ratio_edt
        self.size_thr_edt       =  size_thr_edt

        if mtx_tot.sum() == 0:
            self.framepp1.setImage(self.labbs, levels=(0, self.labbs.max()))
            self.framepp1.setColorMap(pg.ColorMap(np.linspace(0, 1, self.labbs.max()), color=self.colors4map[:self.labbs.max()]))
            self.cut_left_tggl.setEnabled(False)
            self.rmv_wr_tggl.setEnabled(False)
            self.pickcolor_tggl.setEnabled(False)
            self.givecolor_tggl.setEnabled(False)

            self.mdone_flag       =  0
            self.modify_seg_flag  =  0
            self.modify_trk_flag  =  0
            self.roi_flag         =  0
            self.cut_left_flag    =  0

        else:
            self.mtx_tot  =  mtx_tot

            self.left4manual  =  (np.sign(labbs) - np.sign(self.mtx_tot)).astype(np.uint16)
            for t in range(labbs.shape[0]):
                self.left4manual[t, :, :]  =  skmr.label(self.left4manual[t, :, :], connectivity=1).astype(np.uint16)
                self.left4manual[t, :, :]  =  skmr.remove_small_objects(self.left4manual[t, :, :], 15)

#             msk  =  np.zeros(self.left4manual.shape, dtype=np.uint16)
#             for t in range(self.left4manual.shape[0]):
#                 msk[t, :, :] = (ndimage.binary_erosion(np.sign(self.left4manual[t, :, :]), iterations=3) + np.sign(self.left4manual[t, :, :])) == 1

            self.left  =  np.zeros(self.left4manual.shape, dtype=np.uint16)
            for t in range(self.left4manual.shape[0]):
                brds          =  np.sign(ndimage.laplace(self.left4manual[t]))
                brds          =  ndimage.binary_dilation(brds)
                self.left[t]  =  brds + 2 * np.sign(self.left4manual[t]) * (1 - brds)

            self.left  *=  np.sign(self.left4manual)

#             self.left  =   np.zeros(self.mtx_tot.shape, dtype=np.uint16)
#             self.left  +=  msk
#             self.left  +=  2 * (np.sign(self.left4manual) - msk)
            self.framepp1.setImage(self.mtx_tot + self.left, levels=(0, self.mtx_tot.max()))
            self.framepp1.setColorMap(pg.ColorMap(np.linspace(0, 1, self.mtx_tot.max()), color=self.colors4map[:self.mtx_tot.max()]))

            self.buff4man  =  np.copy(self.left4manual)
            self.buffmtx   =  np.copy(self.mtx_tot)

            self.mdone_flag       =  1
            self.modify_seg_flag  =  0
            self.modify_trk_flag  =  1
            self.roi_flag         =  0
            self.cut_left_flag    =  0

        self.rmv_wr_flag      =  0
        self.pickcolor_flag   =  0
        self.givecolor_flag   =  0
        self.pickedlab        =  0
        self.c_count          =  0

        self.setLayout(layout)
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle("Modifier Mitosis Tool")

        self.framepp1.setCurrentIndex(self.cif_start)

    def gfilt_water_var(self, text):
        """Set the parameter for the watershed algorithm"""
        self.gfilt_water_value  =  int(text)

    def circ_thr_var(self, text):
        """Set the circularity threshold"""
        self.circ_thr_value  =  float(text)

    def auto_segm(self):
        """Automatically segment nuclei of the current frame"""
        cif  =  self.framepp1.currentIndex
        self.labbs[cif:]  =  NucleiSegmentStackMultiCore.NucleiSegmentStackCoordinator([self.labbs_start[cif:], self.circ_thr_value, self.gfilt_water_value]).nuclei_labels
        self.framepp1.setImage(self.labbs, levels=(0, self.labbs.max()))
        self.framepp1.setColorMap(pg.ColorMap(np.linspace(0, 1, self.labbs.max()), color=self.colors4map[:self.labbs.max()]))
        self.framepp1.setCurrentIndex(cif)

    def dist_singtrck_var(self, text):
        """Set the distance threshold for the single nucleus tracking"""
        self.dist_singtrck_value  =  float(text)

    def overlap_ratio_var(self, text):
        """Set overlapp ratio for tracking"""
        self.overlap_ratio_value  =  float(text)

    def prolif_ratio_var(self, text):
        """Set proliferation ratio"""
        self.prolif_ratio_value  =  float(text)

    def size_thr_var(self, text):
        """Set nucleus size threshold"""
        self.size_thr_value  =  float(text)

    def mmodify_segm(self):
        """Activate the tool for manual corrections"""
        cif  =  self.framepp1.currentIndex
        # hh   =  self.framepp1.view.viewRange()

        if self.mdone_flag == 0:
            pp       =  self.roi.getHandles()
            pp       =  [self.roi.mapToItem(self.framepp1.imageItem, p.pos()) for p in pp]
            end_pts  =  np.array([[int(pp[0].x()), int(pp[0].y())], [int(pp[1].x()), int(pp[1].y())]])

            self.bufflabbs         =  np.copy(self.labbs)
            self.labbs[cif, :, :]  =  LabelsModify.LabelsModify(self.labbs[cif, :, :], end_pts).labels_fin
            self.framepp1.setImage(self.labbs, levels=(0, self.labbs.max()))

        if self.cut_left_flag == 1:
            pp       =  self.roi.getHandles()
            pp       =  [self.roi.mapToItem(self.framepp1.imageItem, p.pos()) for p in pp]
            end_pts  =  np.array([[int(pp[0].x()), int(pp[0].y())], [int(pp[1].x()), int(pp[1].y())]])

            self.buff4man          =  np.copy(self.left4manual)
            self.buffmtx           =  np.copy(self.mtx_tot)
            self.bufflabbs         =  np.copy(self.labbs)
            self.left4manual[cif]  =  LabelsModify.LabelsModify(self.left4manual[cif], end_pts).labels_fin
            self.labbs[cif]        =  LabelsModify.LabelsModify(self.labbs[cif], end_pts).labels_fin

            self.left[cif]  =  np.zeros(self.left4manual[cif].shape, dtype=np.uint16)
            brds            =  np.sign(ndimage.laplace(self.left4manual[cif]))
            brds            =  ndimage.binary_dilation(brds)
            self.left[cif]  =  brds + 2 * np.sign(self.left4manual[cif]) * (1 - brds)
            # for t in range(self.left4manual.shape[0]):
            #     brds          =  np.sign(ndimage.laplace(self.left4manual[t]))
            #     brds          =  ndimage.binary_dilation(brds)
            #     self.left[t]  =  brds + 2 * np.sign(self.left4manual[t]) * (1 - brds)

            self.left  *=  np.sign(self.left4manual)

#             msk  =  np.zeros(self.left4manual.shape, dtype=np.uint16)
#             for t in range(self.left4manual.shape[0]):
#                 msk[t, :, :]  =  (ndimage.binary_erosion(np.sign(self.left4manual[t, :, :]), iterations=3) + np.sign(self.left4manual[t, :, :])) == 1
#
#             self.left  =   np.zeros(self.mtx_tot.shape, dtype=np.uint16)
#             self.left  +=  msk
#             self.left  +=  2 * (np.sign(self.left4manual) - msk)

            self.framepp1.setImage(self.mtx_tot + self.left, levels=(0, self.mtx_tot.max()), autoRange=False)
            self.framepp1.setColorMap(pg.ColorMap(np.linspace(0, 1, self.mtx_tot.max()), color=self.colors4map[:self.mtx_tot.max()]))

#         self.framepp1.removeItem(self.roi)
        self.framepp1.setCurrentIndex(cif)
        # self.framepp1.view.setXRange(hh[0][0], hh[0][1], padding=.0002)
        # self.framepp1.view.setYRange(hh[1][0], hh[1][1], padding=.0002)

    def mdone(self):
        """Automatic tracking during mitosis."""
        self.modify_seg_flag  =  0
        self.modify_trk_flag  =  1
        if self.roi_flag == 1:
            # self.framepp1.removeItem(self.roi)
            self.roi_flag  =  0

        mts_trk           =  MitosisCrossing3.MitosisCrossing3(self.labbs, self.overlap_ratio_value, self.prolif_ratio_value, self.size_thr_value, self.dist_singtrck_value)
        self.mtx_tot      =  (mts_trk.mtx_tot + 2 * np.sign(mts_trk.mtx_tot)).astype(int)
        self.left4manual  =  mts_trk.left4manual.astype(np.uint16)

#         msk  =  np.zeros(self.left4manual.shape, dtype=np.uint16)
#         for t in range(self.left4manual.shape[0]):
#             msk[t, :, :]  =  (ndimage.binary_erosion(np.sign(self.left4manual[t, :, :]), iterations=3) + np.sign(self.left4manual[t, :, :])) == 1
#
#         self.left  =   np.zeros(self.mtx_tot.shape, dtype=np.uint16)
#         self.left  +=  msk
#         self.left  +=  2 * (np.sign(self.left4manual) - msk)
        self.left  =  np.zeros(self.left4manual.shape, dtype=np.uint16)
        for t in range(self.left4manual.shape[0]):
            brds          =  np.sign(ndimage.laplace(self.left4manual[t]))
            brds          =  ndimage.binary_dilation(brds)
            self.left[t]  =  brds + 2 * np.sign(self.left4manual[t]) * (1 - brds)

        self.left  *=  np.sign(self.left4manual)

        self.framepp1.setImage(self.mtx_tot + self.left, levels=(0, self.mtx_tot.max()))
        self.framepp1.setColorMap(pg.ColorMap(np.linspace(0, 1, self.mtx_tot.max()), color=self.colors4map[:self.mtx_tot.max()]))

        self.buff4man   =  np.copy(self.left4manual)
        self.buffmtx    =  np.copy(self.mtx_tot)
        self.bufflabbs  =  np.copy(self.labbs)

        self.mdone_flag  =  1
        self.cut_left_tggl.setEnabled(True)
        self.rmv_wr_tggl.setEnabled(True)
        self.pickcolor_tggl.setEnabled(True)
        self.givecolor_tggl.setEnabled(True)
        self.dist_singtrck_edt.setEnabled(True)
        self.dist_singtrck_lbl.setEnabled(True)
        # self.mdone_btn.setEnabled(False)

    def cut_left(self, state):
        """Activate checkbox to cut left"""
        if state == QtCore.Qt.Checked:
            self.cut_left_flag  =  1
            if self.rmv_wr_flag  ==  1:
                self.rmv_wr_tggl.toggle()
            if self.pickcolor_flag == 1:
                self.pickcolor_tggl.toggle()
            if self.givecolor_flag == 1:
                self.givecolor_tggl.toggle()
        else:
            self.cut_left_flag  =  0

    def remove_wr(self, state):
        """Activate checkbox to remove badly assigned tags"""
        if state == QtCore.Qt.Checked:
            self.rmv_wr_flag  =  1
            if self.cut_left_flag  ==  1:
                self.cut_left_tggl.toggle()
            if self.pickcolor_flag  ==  1:
                self.pickcolor_tggl.toggle()
            if self.givecolor_flag == 1:
                self.givecolor_tggl.toggle()
        else:
            self.rmv_wr_flag  =  0

    def pickcolor(self, state):
        """Activate checkbox to pick a tag"""
        if state == QtCore.Qt.Checked:
            self.pickcolor_flag  =  1
            if self.cut_left_flag  ==  1:
                self.cut_left_tggl.toggle()
            if self.rmv_wr_flag  ==  1:
                self.rmv_wr_tggl.toggle()
            if self.givecolor_flag == 1:
                self.givecolor_tggl.toggle()
        else:
            self.pickcolor_flag  =  0

    def givecolor(self, state):
        """Activate the checkbox to give a tag"""
        if state == QtCore.Qt.Checked:
            self.givecolor_flag  =  1
            if self.cut_left_flag  ==  1:
                self.cut_left_tggl.toggle()
            if self.rmv_wr_flag  ==  1:
                self.rmv_wr_tggl.toggle()
            if self.pickcolor_flag  ==  1:
                self.pickcolor_tggl.toggle()
        else:
            self.givecolor_flag  =  0

    def click(self, event):
        """Definition of the function involving the mouse-click"""
        event.accept()

        if self.rmv_wr_flag + self.pickcolor_flag + self.givecolor_flag == 0:
            pos        =  np.round(event.pos()).astype(int)
            modifiers  =  QtWidgets.QApplication.keyboardModifiers()

            if modifiers  ==  QtCore.Qt.ShiftModifier:

                if self.c_count - 2 * (self.c_count // 2) == 0:
                    self.pos1  =   pos
                else:
                    try:
                        self.framepp1.removeItem(self.roi)
                    except AttributeError:
                        pass

                    self.roi  =  pg.LineSegmentROI([self.pos1, pos], pen='r')
                    self.framepp1.addItem(self.roi)

                self.c_count  +=  1

        if self.rmv_wr_flag == 1:
            pos        =  np.round(event.pos()).astype(int)
            modifiers  =  QtWidgets.QApplication.keyboardModifiers()

            if modifiers  ==  QtCore.Qt.ShiftModifier:
                if self.mtx_tot[self.framepp1.currentIndex, pos[0], pos[1]] != 0:

                    self.buff4man  =  np.copy(self.left4manual)
                    self.buffmtx   =  np.copy(self.mtx_tot)
                    cif            =  self.framepp1.currentIndex
                    # hh             =  self.framepp1.view.viewRange()

                    lab2rem  =  (self.mtx_tot[cif:, :, :] == self.mtx_tot[cif, np.round(pos[0]).astype(int), np.round(pos[1]).astype(int)]).astype(int)

#                     msk  =  np.zeros(lab2rem.shape, dtype=np.uint16)
#                     for t in range(lab2rem.shape[0]):
#                         msk[t, :, :]  =  (ndimage.binary_erosion(np.sign(lab2rem[t, :, :]), iterations=3) + np.sign(lab2rem[t, :, :])) == 1
#
                    self.mtx_tot[cif:]  *=  1 - lab2rem
                    lab2rem_two         =   np.zeros(lab2rem.shape, dtype=np.uint16)
                    for t in range(self.mtx_tot[cif:].shape[0]):
                        lab2rem_two[t]  =  skmr.label(lab2rem[t] * self.labbs[cif + t], connectivity=1).astype(np.uint16)

                    ref_value  =  self.left4manual.max().astype(np.uint16)
                    for k in range(1, lab2rem_two.max() + 1):
                        self.left4manual[cif:]  +=  np.uint16((ref_value + k)) * (lab2rem_two == np.uint16(k))

                    self.left  =  np.zeros(self.left4manual.shape, dtype=np.uint16)
                    for t in range(self.left4manual.shape[0]):
                        brds          =  np.sign(ndimage.laplace(self.left4manual[t]))
                        brds          =  ndimage.binary_dilation(brds)
                        self.left[t]  =  brds + 2 * np.sign(self.left4manual[t]) * (1 - brds)

                    self.left  *=  np.sign(self.left4manual)

#                     self.left[cif:, :, :]  +=  msk
#                     self.left[cif:, :, :]  +=  (2 * (np.sign(lab2rem) - msk)).astype(np.uint16)
                    self.framepp1.setImage(self.mtx_tot + self.left, levels=(0, self.mtx_tot.max()), autoRange=False)
                    self.framepp1.setColorMap(pg.ColorMap(np.linspace(0, 1, self.mtx_tot.max()), color=self.colors4map[:self.mtx_tot.max()]))

                    self.framepp1.setCurrentIndex(cif)
                    # self.framepp1.view.setXRange(hh[0][0], hh[0][1], padding=.0002)
                    # self.framepp1.view.setYRange(hh[1][0], hh[1][1], padding=.0002)

        if self.pickcolor_flag == 1:
            pos        =  np.round(event.pos()).astype(int)
            modifiers  =  QtWidgets.QApplication.keyboardModifiers()

            if modifiers  ==  QtCore.Qt.ShiftModifier:
                self.pickedlab  =  self.mtx_tot[self.framepp1.currentIndex, np.round(pos[0]).astype(int), np.round(pos[1]).astype(int)]

        if self.givecolor_flag == 1 and self.pickedlab != 0:

            pos        =  np.round(event.pos()).astype(int)
            modifiers  =  QtWidgets.QApplication.keyboardModifiers()

            if modifiers  ==  QtCore.Qt.ShiftModifier:
                if self.left4manual[self.framepp1.currentIndex, pos[0], pos[1]] != 0:

                    self.buff4man  =  np.copy(self.left4manual)
                    self.buffmtx   =  np.copy(self.mtx_tot)
                    cif            =  self.framepp1.currentIndex
                    # hh             =  self.framepp1.view.viewRange()

                    idx_ref   =  self.left4manual[cif, pos[0], pos[1]]
                    rgp_ref   =  regionprops((self.left4manual[cif, :, :] == idx_ref).astype(np.uint16))
                    ctrs_ref  =  rgp_ref[0]['centroid']

                    if self.left4manual[cif, np.round(ctrs_ref[0]).astype(int), np.round(ctrs_ref[1]).astype(int)] != 0:
                        nucleus                        =   NucleiConnect4MTool.NucleiConnect4MTool(self.left4manual[cif:, :, :], idx_ref, self.dist_singtrck_value).nucleus.astype(np.uint16)
                        self.left4manual[cif:, :, :]  *=  (1 - nucleus)
                        self.mtx_tot[cif:, :, :]      +=  self.pickedlab * nucleus
                        self.left[cif:, :, :]         *=  (1 - nucleus)

                    else:
                        nucleus                      =   (self.left4manual[cif, :, :] == self.left4manual[cif, np.round(pos[0]).astype(int), np.round(pos[1]).astype(int)]).astype(np.uint16)
                        self.left4manual[cif, :, :]  *=  (1 - nucleus)
                        self.mtx_tot[cif, :, :]      +=  self.pickedlab * nucleus
                        self.left[cif:, :, :]        *=  (1 - nucleus)

                    self.framepp1.setImage(self.mtx_tot + self.left, autoRange=False)
                    self.framepp1.setCurrentIndex(cif)
                    # self.framepp1.view.setXRange(hh[0][0], hh[0][1], padding=.0002)
                    # self.framepp1.view.setYRange(hh[1][0], hh[1][1], padding=.0002)

    def keyPressEvent(self, event):
        """Shortcut definition for the undo and toogles."""

        if event.key() == (Qt.ControlModifier and Qt.Key_Z):

            cif  =  self.framepp1.currentIndex
            # hh   =  self.framepp1.view.viewRange()

            if self.mdone_flag == 0:
                self.labbs  =  np.copy(self.bufflabbs)
                self.framepp1.setImage(self.labbs)

            else:
                self.mtx_tot      =  np.copy(self.buffmtx)
                self.labbs        =  np.copy(self.bufflabbs)
#                 pg.image(self.labbs)
                self.left4manual  =  np.copy(self.buff4man)
                msk               =  np.zeros(self.left4manual.shape)
                for t in range(self.left4manual.shape[0]):
                    msk[t, :, :]  =  (ndimage.binary_erosion(np.sign(self.left4manual[t, :, :]), iterations=3) + np.sign(self.left4manual[t, :, :])) == 1

                self.left   =  np.zeros(self.mtx_tot.shape)
                self.left  +=  msk
                self.left  +=  2 * (np.sign(self.left4manual) - msk)
                self.framepp1.setImage(self.mtx_tot + self.left, levels=(0, self.mtx_tot.max()), autoRange=False)
                self.framepp1.setColorMap(pg.ColorMap(np.linspace(0, 1, self.mtx_tot.max()), color=self.colors4map[:self.mtx_tot.max()]))

            self.framepp1.setCurrentIndex(cif)
            # self.framepp1.view.setXRange(hh[0][0], hh[0][1], padding=.0002)
            # self.framepp1.view.setYRange(hh[1][0], hh[1][1], padding=.0002)

        if event.key() == (QtCore.Qt.ControlModifier and Qt.Key_Delete):
            self.mmodify_segm()

#        modifiers  =  QtWidgets.QApplication.keyboardModifiers()

        if event.key() == Qt.Key_1:
            if self.givecolor_tggl.checkState() == 2:
                self.pickcolor_tggl.setCheckState(2)
            else:
                self.givecolor_tggl.setCheckState(2)

        if event.key() == Qt.Key_2:
            self.rmv_wr_tggl.setCheckState(2)

        if event.key() == Qt.Key_3:
            self.cut_left_tggl.setCheckState(2)

    def checkwork(self):
        """Check that for each tracked spot the number of connected components is
           1 at the biginning, 2 at the end and there are no oscillations in between."""
        reload(EvolutionController)
        cif  =  self.framepp1.currentIndex
        # hh   =  self.framepp1.view.viewRange()
        evc  =  EvolutionController.EvolutionControllerDM(self.mtx_tot, self.labbs)

        for t in range(evc.conc_nuc_wrong.shape[0]):
            evc.conc_nuc_wrong[t]  =  skmr.label(evc.conc_nuc_wrong[t], connectivity=1).astype(np.uint16)

        self.mtx_tot      *=  1 - np.sign(evc.conc_nuc_wrong)
        self.left4manual  +=  evc.conc_nuc_wrong + self.left4manual.max() * evc.conc_nuc_wrong
        msk               =   np.zeros(self.left4manual.shape, dtype=np.uint16)
        for t in range(self.left4manual.shape[0]):
            msk[t, :, :]  =  (ndimage.binary_erosion(np.sign(self.left4manual[t, :, :]), iterations=3) + np.sign(self.left4manual[t, :, :])) == 1

        self.left  =   np.zeros(self.mtx_tot.shape)
        self.left  +=  msk
        self.left  +=  2 * (np.sign(self.left4manual) - msk)
        self.framepp1.setImage(self.mtx_tot + self.left, levels=(0, self.mtx_tot.max()), autoRange=False)
        self.framepp1.setColorMap(pg.ColorMap(np.linspace(0, 1, self.mtx_tot.max()), color=self.colors4map[:self.mtx_tot.max()]))

        self.framepp1.setCurrentIndex(cif)
        # self.framepp1.view.setXRange(hh[0][0], hh[0][1], padding=.0002)
        # self.framepp1.view.setYRange(hh[1][0], hh[1][1], padding=.0002)

    def shuffle_clrs(self):
        """Shuffle colors of the color map"""
        colors_bff  =  self.colors4map[3:]
        np.random.shuffle(colors_bff)
        self.colors4map[3:]  =  colors_bff
        try:
            self.framepp1.setColorMap(pg.ColorMap(np.linspace(0, 1, self.mtx_tot.max()), color=self.colors4map[:self.mtx_tot.max()]))
        except AttributeError:
            self.framepp1.setColorMap(pg.ColorMap(np.linspace(0, 1, self.labbs.max()), color=self.colors4map[:self.labbs.max()]))

    def update_frame2(self):
        """Function to keep the frames synchronized."""
        self.framepp2.setCurrentIndex(self.framepp1.currentIndex)
        self.frame_numb_lbl.setText("frame  "  +  str(self.framepp1.currentIndex))

    def update_frame1(self):
        """Function to keep the frames synchronized."""
        self.framepp1.setCurrentIndex(self.framepp2.currentIndex)
        self.frame_numb_lbl.setText("frame  "  +  str(self.framepp2.currentIndex))

    @QtCore.pyqtSlot()
    def update_mainwindows2(self):
        np.save('tracked_nucs.npy', self.mtx_tot)
        val  =  self.framepp1.currentIndex
        self.procStart.emit(val)


class SpotsAnalyser(QtWidgets.QWidget):
    """Popup tool to check the spot segmentation"""
    def __init__(self, imarray_green, spots_tracked_old):
        QtWidgets.QWidget.__init__(self)

        self.imarray_green      =  imarray_green
        self.spots_tracked_old  =  spots_tracked_old

        self.imarray_green3c              =  np.zeros(np.append(self.imarray_green.shape, 3))
        self.imarray_green3c[:, :, :, 1]  =  self.imarray_green

        frame1  =  pg.ImageView(self, name='Frame1')
        frame1.ui.roiBtn.hide()
        frame1.ui.menuBtn.hide()
        frame1.setImage(self.imarray_green3c)
        frame1.timeLine.sigPositionChanged.connect(self.update_frame2)

        frame2  =  pg.ImageView(self, name='Frame2')
        frame2.ui.roiBtn.hide()
        frame2.ui.menuBtn.hide()
        frame2.setImage(np.sign(self.spots_tracked_old))
        frame2.timeLine.sigPositionChanged.connect(self.update_frame1)
        frame2.view.setXLink('Frame1')
        frame2.view.setYLink('Frame1')

        tabs  =  QtWidgets.QTabWidget()
        tab1  =  QtWidgets.QWidget()
        tab2  =  QtWidgets.QWidget()

        frame1_box  =  QtWidgets.QHBoxLayout()
        frame1_box.addWidget(frame1)

        frame2_box  =  QtWidgets.QHBoxLayout()
        frame2_box.addWidget(frame2)

        tab1.setLayout(frame1_box)
        tab2.setLayout(frame2_box)

        tabs.addTab(tab1, "Raw")
        tabs.addTab(tab2, "Detected")

        tabsld  =  QtWidgets.QVBoxLayout()
        tabsld.addWidget(tabs)

        frame_numb_lbl = QtWidgets.QLabel("frame  " + '0', self)
        frame_numb_lbl.setFixedSize(110, 13)

        frame_numb_box  =  QtWidgets.QHBoxLayout()
        frame_numb_box.addStretch()
        frame_numb_box.addWidget(frame_numb_lbl)

        layout   =  QtWidgets.QVBoxLayout()
        layout.addLayout(tabsld)
        layout.addLayout(frame_numb_box)

        self.frame1          =  frame1
        self.frame2          =  frame2
        self.frame_numb_lbl  =  frame_numb_lbl

        self.tab3_flag  =  0

        self.setLayout(layout)
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle('Visualize Spots Tool')

    def update_frame2(self):
        """Synchronize frame 1 index with the frame 2 index"""
        self.frame2.setCurrentIndex(self.frame1.currentIndex)
        self.frame_numb_lbl.setText("frame  "  +  str(self.frame1.currentIndex))

    def update_frame1(self):
        """Synchronize frame 1 index with the frame 2 index"""
        self.frame1.setCurrentIndex(self.frame2.currentIndex)
        self.frame_numb_lbl.setText("frame  "  +  str(self.frame2.currentIndex))


class MergePartialTool(QtWidgets.QWidget):
    """Popup tool to merge partial analysis results"""
    def __init__(self, folder_path, mm, nuc_active, evol_check, spots_tracked, nucs_spts_ch, soft_version):
        QtWidgets.QWidget.__init__(self)

        ints_am   =  SpotsFeaturesLoader.SpotsFeaturesLoader(folder_path + '/ints_spots_am_coords.npy').spts_mtx
        ints_bm   =  SpotsFeaturesLoader.SpotsFeaturesLoader(folder_path + '/ints_spots_bm_coords.npy').spts_mtx
        ints_dm   =  SpotsFeaturesLoader.SpotsFeaturesLoader(folder_path + '/ints_spots_dm_coords.npy').spts_mtx
        ints_tot  =  ints_bm[:-1]
        ints_tot  =  np.concatenate((ints_tot, ints_dm), axis=0)
        ints_tot  =  np.concatenate((ints_tot, ints_am[1:]), axis=0)

        vol_am   =  SpotsFeaturesLoader.SpotsFeaturesLoader(folder_path + '/vol_spots_am_coords.npy').spts_mtx
        vol_bm   =  SpotsFeaturesLoader.SpotsFeaturesLoader(folder_path + '/vol_spots_bm_coords.npy').spts_mtx
        vol_dm   =  SpotsFeaturesLoader.SpotsFeaturesLoader(folder_path + '/vol_spots_dm_coords.npy').spts_mtx
        vol_tot  =  vol_bm[:-1]
        vol_tot  =  np.concatenate((vol_tot, vol_dm), axis=0)
        vol_tot  =  np.concatenate((vol_tot, vol_am[1:]), axis=0)

        self.folder_path    =  folder_path
        self.mm             =  mm
        self.nuc_active     =  nuc_active
        self.evol_check     =  evol_check
        self.spots_tracked  =  spots_tracked
        self.nucs_spts_ch   =  nucs_spts_ch
        self.soft_version   =  soft_version
        self.ints_tot       =  ints_tot
        self.vol_tot        =  vol_tot

        fnames_edt  =  QtWidgets.QLineEdit(self)
        fnames_edt.setToolTip('Names and paths of the loaded partial analysis')
        fnames_edt.setText(self.folder_path)

        red                        =  np.load(self.folder_path + '/imarray_red_whole.npy')
        green4D                    =  np.load(self.folder_path + '/green4D_whole.npy')
        t_steps, zlen, xlen, ylen  =  green4D.shape
        green                      =  np.zeros((t_steps, xlen, ylen), dtype=np.uint16)
        for t in range(t_steps):
            for x in range(xlen):
                green[t, x, :]  =  green4D[t, :, x, :].max(0)

        tot              =  np.zeros(red.shape + (3,), dtype=np.uint8)
#         tot[:, :, :, 0]  =  red.astype(float) / red.max()
        tot[:, :, :, 0]  =  (255 * (red / red.max())).astype(np.uint8)
        tot[:, :, :, 1]  =  (255 * (green / green.max())).astype(np.uint8)

        first       =   np.copy(self.evol_check.conc_nuc_ok)
        mycmap      =   np.fromfile('mycmap.bin', 'uint16').reshape((10000, 3))
        colors4map  =   []
        for k in range(mycmap.shape[0]):
            colors4map.append(mycmap[k, :])
        colors4map[0]  =  np.array([0, 0, 0])
        colors4map[1]  =  np.array([0, 255, 0])
        colors4map1    =  colors4map[:int(first.max())]
        mycolormap     =  pg.ColorMap(np.linspace(0, 1, first.max()), color=colors4map1)

        tabs  =  QtWidgets.QTabWidget()
        tab1  =  QtWidgets.QWidget()
        tab2  =  QtWidgets.QWidget()
        tab3  =  QtWidgets.QWidget()
        tab4  =  QtWidgets.QWidget()

        framepp1  =  pg.ImageView(self, name='Frame1')
        framepp1.ui.roiBtn.hide()
        framepp1.ui.menuBtn.hide()
        framepp1.setImage(first)
        framepp1.setColorMap(mycolormap)
        framepp1.timeLine.sigPositionChanged.connect(self.update_frame1)

        framepp2  =  pg.ImageView(self)
        framepp2.ui.roiBtn.hide()
        framepp2.ui.menuBtn.hide()
        framepp2.setImage(tot)
        framepp2.timeLine.sigPositionChanged.connect(self.update_frame2)
        framepp2.view.setXLink('Frame1')
        framepp2.view.setYLink('Frame1')

        framepp3  =  pg.ImageView(self)
        framepp3.ui.roiBtn.hide()
        framepp3.ui.menuBtn.hide()
        framepp3.timeLine.sigPositionChanged.connect(self.update_frame3)
        framepp3.view.setXLink('Frame1')
        framepp3.view.setYLink('Frame1')

        framepp4  =  pg.ImageView(self)
        framepp4.ui.roiBtn.hide()
        framepp4.ui.menuBtn.hide()
        framepp4.timeLine.sigPositionChanged.connect(self.update_frame4)
        framepp4.view.setXLink('Frame1')
        framepp4.view.setYLink('Frame1')

        frame1_box  =  QtWidgets.QHBoxLayout()
        frame1_box.addWidget(framepp1)

        frame2_box  =  QtWidgets.QHBoxLayout()
        frame2_box.addWidget(framepp2)

        frame3_box  =  QtWidgets.QHBoxLayout()
        frame3_box.addWidget(framepp3)

        frame4_box  =  QtWidgets.QHBoxLayout()
        frame4_box.addWidget(framepp4)

        tab1.setLayout(frame1_box)
        tab2.setLayout(frame2_box)
        tab3.setLayout(frame3_box)
        tab4.setLayout(frame4_box)

        tabs.addTab(tab1, "Tracked Nuclei")
        tabs.addTab(tab2, "Raw Data")
        tabs.addTab(tab3, "False Colored")
        tabs.addTab(tab4, "Discarded")

        shuffle_colors_btn  =  QtWidgets.QPushButton("Shuffle Colors", self)
        shuffle_colors_btn.clicked.connect(self.shuffle_colors)
        shuffle_colors_btn.setToolTip('Shuffle colors of the colormap')
        shuffle_colors_btn.setFixedSize(110, 25)

        gastru_params_btn  =  QtWidgets.QPushButton("Find Gastr", self)
        gastru_params_btn.clicked.connect(self.gastru_params)
        gastru_params_btn.setToolTip('Load .lsm file to find gastrulation line')
        gastru_params_btn.setFixedSize(110, 25)

        set_steps_btn  =  QtWidgets.QPushButton("Step", self)
        set_steps_btn.clicked.connect(self.set_steps)
        set_steps_btn.setToolTip('Write spatial organized results')
        set_steps_btn.setFixedSize(50, 25)

        step_analysis_edt  =  QtWidgets.QLineEdit(self)
        step_analysis_edt.setToolTip('Set the step for the spatial analysis')
        step_analysis_edt.textChanged[str].connect(self.step_analysis_var)
        step_analysis_edt.setFixedSize(30, 25)

        step_box  =  QtWidgets.QHBoxLayout()
        step_box.addWidget(step_analysis_edt)
        step_box.addWidget(set_steps_btn)
        step_box.addStretch()

        write_spatial_btn  =  QtWidgets.QPushButton("Write Spatial", self)
        write_spatial_btn.clicked.connect(self.write_spatial)
        write_spatial_btn.setToolTip('Write spatial organized results')
        write_spatial_btn.setFixedSize(110, 25)

        fnames_tabs  =  QtWidgets.QVBoxLayout()
        fnames_tabs.addWidget(fnames_edt)
        fnames_tabs.addWidget(tabs)

        save_merged_btn  =  QtWidgets.QPushButton("Save Merged", self)
        save_merged_btn.clicked.connect(self.save_merged)
        save_merged_btn.setToolTip('Write the merged analysis')
        save_merged_btn.setFixedSize(110, 25)

        group_box  =  QtWidgets.QGroupBox("Spatial")
        group_box.setCheckable(True)
        group_box.setChecked(False)
        group_box.setFixedSize(130, 110)

        in_spatial  =  QtWidgets.QVBoxLayout()
        in_spatial.addWidget(gastru_params_btn)
        in_spatial.addLayout(step_box)
        in_spatial.addWidget(write_spatial_btn)

        group_box.setLayout(in_spatial)

        frame_lbl  =  QtWidgets.QLabel("Frame 0", self)
        frame_lbl.setFixedSize(90, 25)

        command  =  QtWidgets.QVBoxLayout()
        command.addWidget(save_merged_btn)
        command.addStretch()
        command.addWidget(group_box)
        command.addStretch()
        command.addWidget(shuffle_colors_btn)
        command.addWidget(frame_lbl)

        layout  =  QtWidgets.QHBoxLayout()
        layout.addLayout(fnames_tabs)
        layout.addLayout(command)

        self.framepp1    =  framepp1
        self.framepp2    =  framepp2
        self.framepp3    =  framepp3
        self.framepp4    =  framepp4
        self.colors4map  =  colors4map

        self.tot        =  tot
        self.first      =  first
        self.frame_lbl  =  frame_lbl

        self.framepp3.setImage(self.nuc_active.nuclei_active3c)
        self.framepp4.setImage(self.evol_check.conc_nuc_wrong, levels=(0, self.evol_check.conc_nuc_wrong.max()))
        colors4map    =  colors4map[:int(self.evol_check.conc_nuc_wrong.max())]
        mycolormap2  =   pg.ColorMap(np.linspace(0, 1, self.evol_check.conc_nuc_wrong.max()), color=colors4map)
        self.framepp4.setColorMap(mycolormap2)
        self.nuc_act    =  self.nuc_active.nuclei_active
        self.nuc_act3c  =  self.nuc_active.nuclei_active3c

        self.setLayout(layout)
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle("Merging Tool")

    def step_analysis_var(self, text):
        """Set the value of the step"""
        self.step_analysis_value  =  int(text)

    def falsecolored(self):
        """Generate false colored moviie"""
        self.framepp3.setImage(self.nuc_active.nuclei_active3c)
        self.framepp4.setImage(self.evol_check.conc_nuc_wrong, levels=(0, self.evol_check.conc_nuc_wrong.max()))
        self.nuc_act    =  np.copy(self.nuc_active.nuclei_active)
        self.nuc_act3c  =  np.copy(self.nuc_active.nuclei_active3c)

    def gastru_params(self):
        """Set gastrulation line"""
        gastru_fname  =  QtWidgets.QFileDialog.getOpenFileNames(None, "Select lsm file (latest of the movie) to detect gastrulation", filter="*.lsm *.tif *.czi")[0]
        if str(gastru_fname[0])[-3:] == 'lsm' or str(gastru_fname[0])[-3:] == 'tif':
            filedata  =  MultiLoadLsmOrTif5D.MultiLoadLsmOrTif5D(gastru_fname, self.nucs_spts_ch)
        if str(gastru_fname[0])[-3:] == 'czi':
            filedata  =  MultiLoadCzi5D.MultiLoadCzi5D(gastru_fname, self.nucs_spts_ch)

        toshow              =  np.zeros(np.append(filedata.imarray_red.shape, 3))
        toshow[:, :, :, 0]  =  filedata.imarray_red
        toshow[:, :, :, 1]  =  filedata.imarray_green
        ww                  =  pg.image(toshow)
        roi                 =  pg.LinearRegionItem(orientation=True, brush=[0, 0, 0, 0])
        ww.addItem(roi)
        self.roi            =  roi

    def save_merged(self):
        """Save the merged partial analysis"""
        reload(SaveMergedResults)
        SaveMergedResults.SaveMergedResults(self.folder_path, self.mm, self.nuc_active, self.evol_check, self.spots_tracked)
        WriteMergedData.WriteMergedData(self.nuc_act3c, self.first, self.folder_path)

    def set_steps(self):
        """Set the region to study and pop-up a sample"""
        self.roi.update()
        self.mu     =  int(self.roi.getRegion()[1])
        self.sigma  =  int(self.step_analysis_value)
        y_len       =  self.nuc_act3c.shape[2]

        self.mask                                                                                    =  np.zeros(self.nuc_act3c.shape, dtype=np.uint8)
        self.mask[:, :, np.max([self.mu - self.sigma, 0]):np.min([self.mu + self.sigma, y_len]), 0]  =  120 * (self.nuc_act3c[:, :, np.max([self.mu - self.sigma, 0]):np.min([self.mu + self.sigma, y_len]), :].sum(3) == 0)
        self.mask[:, :, np.max([self.mu - self.sigma, 0]):np.min([self.mu + self.sigma, y_len]), 1]  =  120 * (self.nuc_act3c[:, :, np.max([self.mu - self.sigma, 0]):np.min([self.mu + self.sigma, y_len]), :].sum(3) == 0)
        self.mask[:, :, np.max([self.mu - self.sigma, 0]):np.min([self.mu + self.sigma, y_len]), 2]  =  120 * (self.nuc_act3c[:, :, np.max([self.mu - self.sigma, 0]):np.min([self.mu + self.sigma, y_len]), :].sum(3) == 0)

        self.mask  +=  self.nuc_act3c
        pg.image(self.mask)

    def write_spatial(self):
        """Write xls file with the results of the merged analysis"""
        WriteMergedData.WriteMergedDataSpatial(self.folder_path, self.nuc_act, self.mm.conc_wild, self.mm.conc_nuc, self.mu - self.sigma, self.mu + self.sigma, self.folder_path, self.mask, self.spots_tracked, self.ints_tot, self.vol_tot, self.soft_version)

    def shuffle_colors(self):
        """Shuffle colors"""
        colors_bff  =  self.colors4map[1:]
        np.random.shuffle(colors_bff)
        self.colors4map[1:]  =  colors_bff
        mycmap  =  pg.ColorMap(np.linspace(0, 1, self.first.max()), color=self.colors4map)
        self.framepp1.setColorMap(mycmap)

        colors_bff  =  self.colors4map[1:]
        np.random.shuffle(colors_bff)
        self.colors4map[1:]  =  colors_bff
        mycolormap2          =  pg.ColorMap(np.linspace(0, 1, self.evol_check.conc_nuc_wrong.max()), color=self.colors4map)
        self.framepp4.setColorMap(mycolormap2)

    def update_frame1(self):
        """Keep all the frames synchronized from frame 1"""
        self.framepp2.setCurrentIndex(self.framepp1.currentIndex)
        self.framepp3.setCurrentIndex(self.framepp1.currentIndex)
        self.framepp4.setCurrentIndex(self.framepp1.currentIndex)
        self.frame_lbl.setText("Frame " + str(self.framepp1.currentIndex))

    def update_frame2(self):
        """Keep all the frames synchronized from frame 2"""
        self.framepp1.setCurrentIndex(self.framepp2.currentIndex)
        self.framepp3.setCurrentIndex(self.framepp2.currentIndex)
        self.framepp4.setCurrentIndex(self.framepp2.currentIndex)

    def update_frame3(self):
        """Keep all the frames synchronized from frame 3"""
        self.framepp1.setCurrentIndex(self.framepp3.currentIndex)
        self.framepp2.setCurrentIndex(self.framepp3.currentIndex)
        self.framepp4.setCurrentIndex(self.framepp3.currentIndex)

    def update_frame4(self):
        """Keep all the frames synchronized from frame 4"""
        self.framepp1.setCurrentIndex(self.framepp4.currentIndex)
        self.framepp2.setCurrentIndex(self.framepp4.currentIndex)
        self.framepp3.setCurrentIndex(self.framepp4.currentIndex)

    def closeEvent(self, event):
        """Confirm dialogue to close pop-up tool"""
        quit_msg  =  "Are you sure you want to exit the program?"
        reply     =  QtWidgets.QMessageBox.question(self, 'Message', quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


class FlagSpotsDetection(QtWidgets.QDialog):
    """Dialog to input the flag filter/no filter of the spots"""
    def __init__(self, parent=None):
        super(FlagSpotsDetection, self).__init__(parent)

        title_lbl  =  QtWidgets.QLabel("Choose algorithm for spots discrimination", self)
        title_lbl.setFixedSize(300, 25)

        oldnew_combo = QtWidgets.QComboBox(self)
        oldnew_combo.addItem("Filter")
        oldnew_combo.addItem("No Filter")
        oldnew_combo.activated[str].connect(self.oldnew_val)
        oldnew_combo.setFixedSize(300, 25)

        cierra_btn  =  QtWidgets.QPushButton("Ok", self)
        cierra_btn.clicked.connect(self.cierra)
        cierra_btn.setFixedSize(300, 25)

        self.filter_nofilter  =  "Filter"

        layout  =  QtWidgets.QVBoxLayout()
        layout.addWidget(title_lbl)
        layout.addWidget(oldnew_combo)
        layout.addWidget(cierra_btn)

        self.setWindowModality(Qt.ApplicationModal)
        self.setLayout(layout)
        self.setGeometry(300, 300, 350, 100)
        self.setWindowTitle("Choose Algorithm")

    def oldnew_val(self, text):
        self.filter_nofilter  =  text

    def cierra(self):
        self.close()

    def oldnew(self):
        return self.filter_nofilter

    @staticmethod
    def getOldNewFlag(parent=None):
        dialog  =  FlagSpotsDetection(parent)
        result  =  dialog.exec_()
        flag    =  str(dialog.oldnew())
        return flag


class CroppingTool(QtWidgets.QWidget):
    """Popup tool to crop the raw data"""
    procStart  =  QtCore.pyqtSignal(int)

    def __init__(self, imarray_red, imarray_green):
        QtWidgets.QWidget.__init__(self)

        imarray_tot  =  np.zeros(np.append(imarray_red.shape, 3))
        imarray_tot[:, :, :, 0]  =  imarray_red
        imarray_tot[:, :, :, 1]  =  imarray_green

        framepp1  =  pg.ImageView(self)
        framepp1.ui.roiBtn.hide()
        framepp1.ui.menuBtn.hide()
        framepp1.setImage(imarray_tot)

        blue_pen  =  pg.mkPen(color='b', width=2)

        roi  =  pg.RectROI([20, 20], [20, 20], pen=blue_pen)
        framepp1.addItem(roi)

        send_crop_btn  =  QtWidgets.QPushButton("Crop", self)
        send_crop_btn.setFixedSize(120, 25)
        send_crop_btn.clicked.connect(self.crop_to_mainwindows)

        keys  =  QtWidgets.QHBoxLayout()
        keys.addStretch()
        keys.addWidget(send_crop_btn)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addWidget(framepp1)
        layout.addLayout(keys)

        self.framepp1  =  framepp1
        self.roi       =  roi

        self.setLayout(layout)
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle("Crop Tool")

    @QtCore.pyqtSlot()
    def crop_to_mainwindows(self):
        val  =  self.framepp1.currentIndex
        self.procStart.emit(val)


class SpotNcuDistanceThr(QtWidgets.QDialog):
    """Popup tool to set spot-nucleus maximum distance"""
    def __init__(self, parent=None):
        super(SpotNcuDistanceThr, self).__init__(parent)

        numb_pixels_lbl  =  QtWidgets.QLabel("Numb of pixels", self)
        numb_pixels_lbl.setFixedSize(110, 25)

        numb_pixels_edt = QtWidgets.QLineEdit(self)
        numb_pixels_edt.setToolTip("Max distance in pixels")
        numb_pixels_edt.setFixedSize(30, 22)
        numb_pixels_edt.textChanged[str].connect(self.numb_pixels_var)

        input_close_btn  =  QtWidgets.QPushButton("Ok", self)
        input_close_btn.clicked.connect(self.input_close)
        input_close_btn.setToolTip('Input values')
        input_close_btn.setFixedSize(50, 25)

        numb_pixels_lbl_edit_box  =  QtWidgets.QHBoxLayout()
        numb_pixels_lbl_edit_box.addWidget(numb_pixels_lbl)
        numb_pixels_lbl_edit_box.addWidget(numb_pixels_edt)

        input_close_box  =  QtWidgets.QHBoxLayout()
        input_close_box.addStretch()
        input_close_box.addWidget(input_close_btn)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addLayout(numb_pixels_lbl_edit_box)
        layout.addLayout(input_close_box)

        self.setWindowModality(Qt.ApplicationModal)
        self.setLayout(layout)
        self.setGeometry(300, 300, 200, 50)
        self.setWindowTitle("Spot Nuc Max Distance")

    def numb_pixels_var(self, text):
        self.numb_pixels_value  =  int(text)

    def input_close(self):
        self.close()

    def numb_pixels(self):
        return self.numb_pixels_value

    @staticmethod
    def getNumb(parent=None):
        dialog       =  SpotNcuDistanceThr(parent)
        result       =  dialog.exec_()
        numb_pixels  =  dialog.numb_pixels()
        return numb_pixels


class TileMap(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(TileMap, self).__init__(parent)

        numb_hor_squares_lbl  =  QtWidgets.QLabel("Numb of Hor Squares", self)
        numb_hor_squares_lbl.setFixedSize(150, 25)

        numb_hor_squares_edt  =  QtWidgets.QLineEdit(self)
        numb_hor_squares_edt.setToolTip("Number of horizontal squares you have in your tile scan")
        numb_hor_squares_edt.setFixedSize(30, 22)
        numb_hor_squares_edt.textChanged[str].connect(self.numb_hor_squares_var)

        capture_hor_square_lbl  =  QtWidgets.QLabel("Capture Square", self)
        capture_hor_square_lbl.setFixedSize(150, 25)

        capture_hor_square_edt  =  QtWidgets.QLineEdit(self)
        capture_hor_square_edt.setToolTip("Horizontal position of the capture square (1 means the first from left)")
        capture_hor_square_edt.setFixedSize(30, 22)
        capture_hor_square_edt.textChanged[str].connect(self.capture_hor_square_var)

        numb_ver_squares_lbl  =  QtWidgets.QLabel("Numb of Ver Squares", self)
        numb_ver_squares_lbl.setFixedSize(150, 25)

        numb_ver_squares_edt  =  QtWidgets.QLineEdit(self)
        numb_ver_squares_edt.setToolTip("Number of vertical squares you have in your tile scan")
        numb_ver_squares_edt.setFixedSize(30, 22)
        numb_ver_squares_edt.textChanged[str].connect(self.numb_ver_squares_var)

        capture_ver_square_lbl  =  QtWidgets.QLabel("Capture Square", self)
        capture_ver_square_lbl.setFixedSize(150, 25)

        capture_ver_square_edt  =  QtWidgets.QLineEdit(self)
        capture_ver_square_edt.setToolTip("Vertical position of the capture square (1 means the first from the top)")
        capture_ver_square_edt.setFixedSize(30, 22)
        capture_ver_square_edt.textChanged[str].connect(self.capture_ver_square_var)

        input_close_btn  =  QtWidgets.QPushButton("Ok", self)
        input_close_btn.clicked.connect(self.input_close)
        input_close_btn.setToolTip('Input values')
        input_close_btn.setFixedSize(50, 25)

        numb_hor_squares_lbl_edit_box  =  QtWidgets.QHBoxLayout()
        numb_hor_squares_lbl_edit_box.addWidget(numb_hor_squares_lbl)
        numb_hor_squares_lbl_edit_box.addWidget(numb_hor_squares_edt)

        capture_hor_square_lbl_edit_box  =  QtWidgets.QHBoxLayout()
        capture_hor_square_lbl_edit_box.addWidget(capture_hor_square_lbl)
        capture_hor_square_lbl_edit_box.addWidget(capture_hor_square_edt)

        numb_ver_squares_lbl_edit_box  =  QtWidgets.QHBoxLayout()
        numb_ver_squares_lbl_edit_box.addWidget(numb_ver_squares_lbl)
        numb_ver_squares_lbl_edit_box.addWidget(numb_ver_squares_edt)

        capture_ver_square_lbl_edit_box  =  QtWidgets.QHBoxLayout()
        capture_ver_square_lbl_edit_box.addWidget(capture_ver_square_lbl)
        capture_ver_square_lbl_edit_box.addWidget(capture_ver_square_edt)

        input_close_box  =  QtWidgets.QHBoxLayout()
        input_close_box.addStretch()
        input_close_box.addWidget(input_close_btn)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addLayout(numb_hor_squares_lbl_edit_box)
        layout.addLayout(capture_hor_square_lbl_edit_box)
        layout.addLayout(numb_ver_squares_lbl_edit_box)
        layout.addLayout(capture_ver_square_lbl_edit_box)
        layout.addLayout(input_close_box)

        self.setWindowModality(Qt.ApplicationModal)
        self.setLayout(layout)
        self.setGeometry(300, 300, 300, 100)
        self.setWindowTitle("Tile Map Coordinates")

    def numb_hor_squares_var(self, text):
        self.numb_hor_squares_value  =  int(text)

    def capture_hor_square_var(self, text):
        self.capture_hor_square_value  =  int(text)

    def numb_ver_squares_var(self, text):
        self.numb_ver_squares_value  =  int(text)

    def capture_ver_square_var(self, text):
        self.capture_ver_square_value  =  int(text)

    def input_close(self):
        self.close()

    def tile_info(self):
        return [self.numb_hor_squares_value, self.capture_hor_square_value, self.numb_ver_squares_value, self.capture_ver_square_value]

    @staticmethod
    def getNumb(parent=None):
        dialog     =  TileMap(parent)
        result     =  dialog.exec_()
        tile_info  =  dialog.tile_info()
        return tile_info


class SetColorChannel(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(SetColorChannel, self).__init__(parent)

        nuclei_channel_lbl  =  QtWidgets.QLabel("Nuclei Channel", self)
        nuclei_channel_lbl.setFixedSize(100, 22)

        spots_channel_lbl  =  QtWidgets.QLabel("Spots Channel", self)
        spots_channel_lbl.setFixedSize(100, 22)

        nuclei_channel_combo  =  QtWidgets.QComboBox(self)
        nuclei_channel_combo.addItem("1")
        nuclei_channel_combo.addItem("2")
        nuclei_channel_combo.addItem("3")
        nuclei_channel_combo.activated[str].connect(self.nuclei_channel_switch)
        nuclei_channel_combo.setCurrentIndex(1)
        nuclei_channel_combo.setFixedSize(45, 25)

        spots_channel_combo  =  QtWidgets.QComboBox(self)
        spots_channel_combo.addItem("1")
        spots_channel_combo.addItem("2")
        spots_channel_combo.addItem("3")
        spots_channel_combo.activated[str].connect(self.spots_channel_switch)
        spots_channel_combo.setCurrentIndex(0)
        spots_channel_combo.setFixedSize(45, 25)

        enter_values_btn  =  QtWidgets.QPushButton("OK", self)
        enter_values_btn.setToolTip('Set Channels Number')
        enter_values_btn.setFixedSize(60, 25)
        enter_values_btn.clicked.connect(self.enter_values)

        nuclei_box  =  QtWidgets.QHBoxLayout()
        nuclei_box.addWidget(nuclei_channel_lbl)
        nuclei_box.addWidget(nuclei_channel_combo)

        spots_box  =  QtWidgets.QHBoxLayout()
        spots_box.addWidget(spots_channel_lbl)
        spots_box.addWidget(spots_channel_combo)

        enter_box  =  QtWidgets.QHBoxLayout()
        enter_box.addStretch()
        enter_box.addWidget(enter_values_btn)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addLayout(nuclei_box)
        layout.addLayout(spots_box)
        layout.addLayout(enter_box)

        self.nuclei_channel  =  2
        self.spots_channel   =  1

        self.setWindowModality(Qt.ApplicationModal)
        self.setLayout(layout)
        self.setGeometry(300, 300, 200, 100)
        self.setWindowTitle("Set Channels")

    def nuclei_channel_switch(self, text):
        self.nuclei_channel  =  int(text)

    def spots_channel_switch(self, text):
        self.spots_channel  =  int(text)

    def enter_values(self):
        self.channels_values  =  np.array([self.nuclei_channel, self.spots_channel])
        self.close()

    def params(self):
        return self.channels_values

    @staticmethod
    def getChannels(parent=None):
        dialog  =  SetColorChannel(parent)
        result  =  dialog.exec_()
        flag    =  dialog.params()
        return flag


class RemoveMitoticalSpots(QtWidgets.QWidget):
    procStart  =  QtCore.pyqtSignal()
    """Pop up tool to remove mitotical Spots"""
    def __init__(self, filedata, spots_3D):
        QtWidgets.QWidget.__init__(self)

        self.imarray_red          =  filedata.imarray_red
        self.imarray_green        =  filedata.imarray_green
        self.green4D              =  filedata.green4D
        self.spots_3D             =  spots_3D

        raw2chs              =  np.zeros(self.imarray_red.shape + (3,))
        raw2chs[:, :, :, 0]  =  self.imarray_red / 2
        raw2chs[:, :, :, 1]  =  self.imarray_green

        framepp_3chs  =  pg.ImageView(self, name="Frame_3chs")
        framepp_3chs.ui.roiBtn.hide()
        framepp_3chs.ui.menuBtn.hide()
        framepp_3chs.setImage(raw2chs)
        framepp_3chs.timeLine.sigPositionChanged.connect(self.update_frame_segtrk)

        framepp_segtrk  =  pg.ImageView(self, name="framepp_seg_trk")
        framepp_segtrk.ui.roiBtn.hide()
        framepp_segtrk.ui.menuBtn.hide()
        framepp_segtrk.setImage(np.zeros(self.imarray_green.shape))
        framepp_segtrk.timeLine.sigPositionChanged.connect(self.update_frame_3chs)

        framepp_3chs.view.setXLink("framepp_seg_trk")
        framepp_3chs.view.setYLink("framepp_seg_trk")

        tabs_left   =  QtWidgets.QTabWidget()
        tab_3chs    =  QtWidgets.QWidget()
        tab_segtrk  =  QtWidgets.QWidget()

        frame_3chs_box  =  QtWidgets.QHBoxLayout()
        frame_3chs_box.addWidget(framepp_3chs)

        frame_segtrk_box  =  QtWidgets.QHBoxLayout()
        frame_segtrk_box.addWidget(framepp_segtrk)

        tab_3chs.setLayout(frame_3chs_box)
        tab_segtrk.setLayout(frame_segtrk_box)

        tabs_left.addTab(tab_3chs, "Raw 3 chs")
        tabs_left.addTab(tab_segtrk, "Segm or Track")

        frame_lbl  =  QtWidgets.QLabel(self)
        frame_lbl.setFixedSize(65, 25)
        frame_lbl.setText("Frame 0")

        frame_chosen_lbl  =  QtWidgets.QLabel(self)
        frame_chosen_lbl.setFixedSize(35, 25)
        frame_chosen_lbl.setText("(0)")
        frame_chosen_lbl.setStyleSheet('color: red')

        frame_box  =  QtWidgets.QHBoxLayout()
        frame_box.addWidget(frame_lbl)
        frame_box.addWidget(frame_chosen_lbl)

        t_track_end_btn  =  QtWidgets.QPushButton("T End", self)
        t_track_end_btn.clicked.connect(self.t_track_end)
        t_track_end_btn.setToolTip('Spots appearing up to this frframe_3chs_box will be tracked and removd')
        t_track_end_btn.setFixedSize(90, 25)

        frame_lbl_tstart_box  =  QtWidgets.QVBoxLayout()
        frame_lbl_tstart_box.addLayout(frame_box)
        frame_lbl_tstart_box.addWidget(t_track_end_btn)

        dist_thr_lbl  =  QtWidgets.QLabel(self)
        dist_thr_lbl.setFixedSize(51, 25)
        dist_thr_lbl.setText("Dist Thr")

        dist_thr_edt  =  QtWidgets.QLineEdit(self)
        dist_thr_edt.textChanged[str].connect(self.dist_thr_var)
        dist_thr_edt.setToolTip("Set the maximum distance between two consecutive positions of a spot (suggested value 15)")
        dist_thr_edt.setFixedSize(45, 25)

        dist_thr_box  =  QtWidgets.QHBoxLayout()
        dist_thr_box.addWidget(dist_thr_lbl)
        dist_thr_box.addWidget(dist_thr_edt)

        track_btn  =  QtWidgets.QPushButton("Track", self)
        track_btn.clicked.connect(self.track)
        track_btn.setToolTip('Track Spots')
        track_btn.setFixedSize(110, 25)

        track_box  =  QtWidgets.QVBoxLayout()
        track_box.addLayout(dist_thr_box)
        track_box.addWidget(track_btn)

        vert_line_two  =  QtWidgets.QFrame()
        vert_line_two.setFrameStyle(QtWidgets.QFrame.VLine)

        vert_line_three  =  QtWidgets.QFrame()
        vert_line_three.setFrameStyle(QtWidgets.QFrame.VLine)

        send_btn  =  QtWidgets.QPushButton("Send", self)
        send_btn.clicked.connect(self.send)
        send_btn.setToolTip('Send results to the main GUI')
        send_btn.setFixedSize(110, 25)

        commands_box  =  QtWidgets.QHBoxLayout()
        commands_box.addLayout(frame_lbl_tstart_box)
        commands_box.addWidget(vert_line_two)
        commands_box.addLayout(track_box)
        commands_box.addWidget(vert_line_three)
        commands_box.addStretch()
        commands_box.addWidget(send_btn)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addWidget(tabs_left)
        layout.addLayout(commands_box)

        self.setLayout(layout)
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle("Remove Mitotical TS")

        self.framepp_3chs      =  framepp_3chs
        self.framepp_segtrk    =  framepp_segtrk
        self.frame_lbl         =  frame_lbl
        self.frame_chosen_lbl  =  frame_chosen_lbl

        self.t_track_end_value  =  0
        self.framepp_segtrk.setImage(np.sign(self.spots_3D.spots_vol))

    def update_frame_3chs(self):
        self.framepp_3chs.setCurrentIndex(self.framepp_segtrk.currentIndex)
        self.frame_lbl.setText("Frame " + str(self.framepp_3chs.currentIndex))

    def update_frame_segtrk(self):
        self.framepp_segtrk.setCurrentIndex(self.framepp_3chs.currentIndex)

    def t_track_end(self):
        self.t_track_end_value  =  int(self.framepp_3chs.currentIndex)
        self.frame_chosen_lbl.setText("(" + str(self.t_track_end_value) + ")")

    def spts_thr_var(self, text):
        self.spts_thr_value  =  float(text)

    def dist_thr_var(self, text):
        self.dist_thr_value  =  float(text)

    def track(self):
        reload(VisualTracked)
        reload(SpotTracker)
        self.spts_trck_info  =  SpotTracker.SpotTracker(np.copy(self.spots_3D.spots_tzxy), self.dist_thr_value, self.t_track_end_value).spts_trck_info
        bff2show             =  VisualTracked.VisualTracked(self.spots_3D.spots_coords, self.spts_trck_info, self.spots_3D.spots_vol, self.green4D.shape).visual_tracked
        self.spots_trkd      =  bff2show == 2
        self.framepp_segtrk.setImage(bff2show)
        self.framepp_segtrk.updateImage()

    @QtCore.pyqtSlot()
    def send(self):
        self.spots_3d  =  VisualTracked.RemoveTracked(self.spots_3D.spots_coords, self.spots_3D.spots_tzxy, self.spots_3D.spots_ints, self.spots_3D.spots_vol, self.spts_trck_info, self.green4D)

        self.procStart.emit()


class MessageSpotsCoords(QtWidgets.QWidget):
    """Choose which part of the embyo to analyse"""
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self)

        choose_lbl  =  QtWidgets.QLabel("Run Spots Detection in the main GUI before", self)
        choose_lbl.setFixedSize(340, 22)

        ok_btn  =  QtWidgets.QPushButton("Ok", self)
        ok_btn.setFixedSize(120, 25)
        ok_btn.clicked.connect(self.chiudi)

        btn_box  =  QtWidgets.QHBoxLayout()
        btn_box.addStretch()
        btn_box.addWidget(ok_btn)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addWidget(choose_lbl)
        layout.addLayout(btn_box)

        self.setWindowModality(Qt.ApplicationModal)
        self.setLayout(layout)
        self.setGeometry(300, 300, 360, 25)
        self.setWindowTitle("Nucs-Cito-Embryo")
        self.show()

    def chiudi(self):
        """Close dialog"""
        self.close()


class CheckResults(QtWidgets.QWidget):
    """Popup tool to check the analyse"""
    def __init__(self, folder_name):
        QtWidgets.QWidget.__init__(self)

        nuc_active3c   =  LoadMergedResults.LoadMergedResultsNucActive(folder_name).nuclei_active3c

        frame_clrs  =  pg.ImageView()
        frame_clrs.ui.menuBtn.hide()
        frame_clrs.ui.roiBtn.hide()
        frame_clrs.ui.histogram.hide()
        frame_clrs.setImage(nuc_active3c)
        frame_clrs.timeLine.sigPositionChanged.connect(self.img_index_change)

        tag_nuc_edt  =  QtWidgets.QLineEdit()
        tag_nuc_edt.setFixedSize(45, 25)
        tag_nuc_edt.setToolTip("Set the tag of the nucleus you want to follow")
        tag_nuc_edt.textChanged[str].connect(self.tag_nuc_var)
        tag_nuc_edt.returnPressed.connect(self.tag_nuc_show)

        tag_nuc_lbl  =  QtWidgets.QLabel("Nuc_id ")
        tag_nuc_lbl.setFixedSize(50, 25)

        frame_numb_lbl  =  QtWidgets.QLabel("frame numb ")
        frame_numb_lbl.setFixedSize(100, 25)
        frame_numb_lbl.setToolTip("Number of the current frame")

        tag_edt_lbl_box  =  QtWidgets.QHBoxLayout()
        tag_edt_lbl_box.addWidget(tag_nuc_lbl)
        tag_edt_lbl_box.addWidget(tag_nuc_edt)

        commands_box  =  QtWidgets.QVBoxLayout()
        commands_box.addLayout(tag_edt_lbl_box)
        commands_box.addStretch()
        commands_box.addWidget(frame_numb_lbl)

        layout  =  QtWidgets.QHBoxLayout()
        layout.addWidget(frame_clrs)
        layout.addLayout(commands_box)

        self.frame_numb_lbl  =  frame_numb_lbl
        self.frame_clrs      =  frame_clrs
        self.nuc_active3c    =  nuc_active3c
        self.mm              =  LoadMergedResults.LoadMergedResults_mm(folder_name)
        self.spots_tracked   =  SpotsFeaturesLoader.SpotsFeaturesLoader(folder_name + '/spots_tracked_coords.npy').spts_mtx

        self.setLayout(layout)
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle("Remove Mitotical TS")

    def tag_nuc_var(self, text):
        """Set the tag of nucleus to check"""
        self.tag_nuc_value  =  int(text)

    def tag_nuc_show(self):
        """Prepare the matrix image to show"""
        mtx2show               =  np.copy(self.nuc_active3c / 4).astype(np.uint8)
        cal                    =  (((self.mm.conc_nuc == self.tag_nuc_value) + (self.spots_tracked == self.tag_nuc_value))).astype(np.uint8)
        mtx2show[:, :, :, 0]  +=  3 * mtx2show[:, :, :, 0] * cal
        mtx2show[:, :, :, 1]  +=  3 * mtx2show[:, :, :, 1] * cal
        mtx2show[:, :, :, 2]  +=  3 * mtx2show[:, :, :, 2] * cal
        cif                    =  self.frame_clrs.currentIndex
        self.frame_clrs.setImage(mtx2show)
        self.frame_clrs.setCurrentIndex(cif)

    def img_index_change(self):
        """Update the frame number label"""
        self.frame_numb_lbl.setText("Frame numb " + str(self.frame_clrs.currentIndex))


def main():
    app         =  QtWidgets.QApplication(sys.argv)
    splash_pix  =  QtGui.QPixmap('Icons/van.png')
    splash      =  QtWidgets.QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()
    app.processEvents()
    ex   =  MainWindow()
    splash.finish(ex)
    sys.exit(app.exec_())


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':

    main()
    sys.excepthook = except_hook
