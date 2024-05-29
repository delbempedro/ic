import classes
import pyqtgraph as pg
import sys

from pyqtgraph.Qt.QtWidgets import *
from pyqtgraph.Qt.QtGui import *
from pyqtgraph.Qt.QtCore import *

selector = classes.selector()
selector.change_mode("angle")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.win = pg.GraphicsLayoutWidget(show=True, size=(1000,800), border=True)
        self.win.setGeometry(100, 100, 1000, 500)
        self.last_point = QPoint()
        self.current_point = QPoint()
        self.shapes = []

        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = self.current_point
            self.point1 = classes.point(self.last_point.x(), self.last_point.y())

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.current_point = event.pos()
            self.point2 = classes.point(self.current_point.x(), self.current_point.y())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            angle = selector.create_object(self.point1, self.point2)
            a = pg.PolyLineROI([self.point1.x, self.point1.y, self.point2.x, self.point2.y], closed=True)
            self.update()


app = pg.mkQApp("anglebutton")
win = MainWindow()
win.show()

sys.exit(app.exec_())

if __name__ == "__main__":
    pg.exec()