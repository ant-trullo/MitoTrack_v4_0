"""This function finds the atero-posterior coordinate of your stack.

Given tile scan image plus tile information as input, this function founds the
the ateroposterior coordinate of your time series in the embryo. Tile information
are the number of horizonatl and vertical squares in which the tile is divided
and the coordinate in this frame of the square in which you aquired. As a result
a .txt file will be written with this info and a .png image with the tile scan
and a red square for the aquisition frame.
"""


import numpy as np
import pyqtgraph as pg
import tifffile
import czifile
from scipy import ndimage
from skimage.filters import threshold_otsu
import pyqtgraph.exporters



class FromTile2GlobCoordinate:
    def __init__(self, tile_fname, tile_info, txt_path):

        if tile_fname[-4:] == '.lsm' or tile_fname[-4:] == '.tif':
            tile_img  =  tifffile.imread(str(tile_fname))[0, 0, 1, :, :]                  # load and re-orient tile image
            tile_img  =  np.rot90(tile_img, k=3)

        if tile_fname[-4:] == '.czi':
            tile_img  =  np.squeeze(czifile.imread(tile_fname))[1, :, :]
            tile_img  =  np.rot90(tile_img, k=3)

        tile_img_f  =  ndimage.filters.gaussian_filter(tile_img, 10)                    # gaussian filter with very high kernel to find the calc of the embryo (after Otsu thresholding)
        val         =  threshold_otsu(tile_img_f)
        mask        =  (tile_img_f > val)
        pts         =  np.where(mask.sum(1) != 0)[0]
        start_pts   =  pts[0]
        length_pts  =  pts[-1] - pts[0]                                                 # coordinate of the extrema of the embryo

        square_x_pix_size  =  tile_img.shape[0] / tile_info[0]
        square_y_pix_size  =  tile_img.shape[1] / tile_info[2]                          # size of squares in pixels

        perc_first_pts    =  ((tile_info[1] - 1) * square_x_pix_size - start_pts) * 100 / length_pts                    # coordinate o percentage of the aquisition square
        perc_last_pts     =  (tile_info[1] * square_x_pix_size - start_pts) * 100 / length_pts
        x_coord_cntr_img  =  np.linspace(perc_first_pts, perc_last_pts, square_x_pix_size)

        w        =  pg.image(tile_img)                                                                                  # write the png image

        red_pen  =  pg.mkPen(color='r', width=2)
        roi      =  pg.RectROI([square_x_pix_size * (tile_info[1] - 1), square_y_pix_size * (tile_info[3] - 1)], [square_x_pix_size, square_y_pix_size], pen=red_pen)
        w.addItem(roi)
        roi.removeHandle(0)

        exporter  =  pyqtgraph.exporters.ImageExporter(w.view)
        a  =  str(txt_path)
        exporter.export(a[:-a[::-1].find('/')] + 'TileScanMap.png')

        filetxt  =  open(txt_path, "w")                                                                     # write the .txt file
        filetxt.write(tile_fname  + "\n")
        filetxt.write("Start coordinate:  " + str(x_coord_cntr_img[0])  + "\n")
        filetxt.write("End coordinate:    " + str(x_coord_cntr_img[-1]) + "\n")
        filetxt.close()
