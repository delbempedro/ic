import sys

import pyqtgraph as pg

from pyqtgraph.Qt.QtWidgets import QApplication, QMainWindow, QWidget
from pyqtgraph.Qt.QtGui import QPainter, QPen
from pyqtgraph.Qt.QtCore import Qt, QPoint

app2 = pg.mkQApp("ColorButton Example")
win2 = QMainWindow()
btn = pg.ColorButton()
win2.setCentralWidget(btn)
win2.show()
win2.setWindowTitle('pyqtgraph example: ColorButton')

def change(btn):
    print("change", btn.color())
def done(btn):
    print("done", btn.color())

class DrawingWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Drawing Wndow")
        self.setGeometry(100, 100, 1000, 500)

        self.canvas = CanvasWidget(self)
        self.setCentralWidget(self.canvas)
        
class CanvasWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setGeometry(0, 200, 1000, 500)

        self.drawing = False
        self.last_point = QPoint()
        self.current_point = QPoint()
        self.shapes = []

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.drawing == False:
                self.drawing = True
                self.last_point = event.pos()
                self.current_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.current_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            shape = (self.last_point, self.current_point)
            self.shapes.append(shape)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(btn.color())
        painter.setPen(pen)

        for shape in self.shapes:
            painter.drawLine(shape[0], shape[1])

        if self.drawing:
            painter.drawLine(self.last_point, self.current_point)

def main():
    app = QApplication(sys.argv)
    win = DrawingWindow()
    win.show()

    btn.sigColorChanging.connect(change)
    btn.sigColorChanged.connect(done)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()