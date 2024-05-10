import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


plot = pg.plot()   ## create an empty plot widget

## Create text object, use HTML tags to specify color/size
text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FFF;">Today Bayern will</span><br><span style="color: #FF0; font-size: 16pt;">WIN</span></div>', anchor=(-0.3,0.5), angle=45, border='w', fill=(0, 0, 255, 100))
plot.addItem(text)
text.setPos(0, 255)

## Draw an arrowhead next to the text box
arrow = pg.ArrowItem(pos=(0, 255), angle=-45)
plot.addItem(arrow)

if __name__ == '__main__':
    pg.exec()