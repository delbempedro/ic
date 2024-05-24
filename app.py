import classes
import pyqtgraph as pg
import sys

from pyqtgraph.Qt.QtWidgets import *
from pyqtgraph.Qt.QtGui import *
from pyqtgraph.Qt.QtCore import *

selector = classes.selector()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.win = pg.GraphicsLayoutWidget(show=True, size=(1000,800), border=True)
        self.win.setGeometry(100, 100, 1000, 500)
        self.last_point = QPoint()
        self.current_point = QPoint()
        self.shapes = []

        self.QPushButton = pg.PathButton("Angle")
        self.QPushButton.setGeometry(0, 0, 100, 30)
        self.QPushButton.clicked.connect(self.clickme)

    def clickme(self): 
        selector.change_mode("angle")
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if selector._mode == "angle":
                self.last_point = self.current_point
                self.current_point = event.pos()
                angle = selector.create_object(self.last_point, self.current_point)
                a = pg.PolyLineROI([[0,0], [10,10], [10,30], [30,10]], closed=True)


app = pg.mkQApp("anglebutton")
win = MainWindow()
win.show()

sys.exit(app.exec_())

if __name__ == "__main__":
    pg.exec()