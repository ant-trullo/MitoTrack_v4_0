"""This function provides only a pg.image frame with frame number updated

in a text.
"""


import pyqtgraph as pg


class PopUpTool:
    def __init__(self, input_args):

        w = pg.image(input_args[0], title=input_args[1])
        
        if len(input_args) == 3:
            w.setColorMap(input_args[2])


        lx = []
        for t in range(input_args[0].shape[0]):
            lx.append(pg.TextItem(str(t), color='r', anchor=(0, 1)))
        w.addItem(lx[0])
        self.current  =  0

        def upadateframe():
            w.removeItem(lx[self.current])
            w.addItem(lx[w.currentIndex])
            self.current  =  w.currentIndex

        w.timeLine.sigPositionChanged.connect(upadateframe)
