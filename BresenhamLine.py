"""Implementation of BresenhamLine algorithm.

Given two points coordinates, this function returns the coordinate of all
the pixels belonging to segment that connects the points.
"""

import numpy as np


class BresenhamLine:

    def __init__(self, x, y, x2, y2):
        """Brensenham line algorithm"""
        self.x   =  x.astype(np.int)
        self.x2  =  x2.astype(np.int)
        self.y   =  y.astype(np.int)
        self.y2  =  y2.astype(np.int)


        steep = 0
        coords = []
        dx = abs(x2 - x)
        dy = abs(y2 - y)
        if (x2 - x) > 0:
            sx = 1
        else:
            sx = -1
            dy = abs(y2 - y)
        if (y2 - y) > 0:
            sy = 1
        else:
            sy = -1
        if dy > dx:
            steep = 1
            x, y    =  y, x
            dx, dy  =  dy, dx
            sx, sy  =  sy, sx
        d = (2 * dy) - dx
        for i in range(0, dx.astype(np.int)):
            if steep:
                coords.append((y, x))
            else:
                coords.append((x, y))
            while d >= 0:
                y  +=  sy
                d  -=  (2 * dx)
            x  +=  sx
            d  += (2 * dy)
        coords.append((x2, y2))

        self.coords  =  coords
