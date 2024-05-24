import numpy as np

import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder='row-major')

## create GUI
app = pg.mkQApp("ROI Examples")
w = pg.GraphicsLayoutWidget(show=True, size=(1000,800), border=True)
w.setWindowTitle('pyqtgraph example: ROI Examples')

w2 = w.addLayout(row=0, col=1)
label2 = w2.addLabel('text', row=0, col=0)
v2a = w2.addViewBox(row=1, col=0, lockAspect=True)
r2a = pg.PolyLineROI([[0,0], [10,10], [10,30], [30,10]], closed=True)
v2a.addItem(r2a)
r2b = pg.PolyLineROI([[0,-20], [10,-10], [10,-30]], closed=False)
v2a.addItem(r2b)
v2a.disableAutoRange('xy')
#v2b.disableAutoRange('xy')
v2a.autoRange()
#v2b.autoRange()

if __name__ == '__main__':
    pg.exec()