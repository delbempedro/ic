import classes
import pyqtgraph as pg
import sys

from pyqtgraph.Qt.QtWidgets import *
from pyqtgraph.Qt.QtGui import *
from pyqtgraph.Qt.QtCore import *

 
# creating a push button
anglebutton = QPushButton("Angle")
 
# setting geometry of button
anglebutton.setGeometry(0, 0, 100, 30)
 
# adding action to a button
anglebutton.clicked.connect(self.clickme)

# creating a push button
selecbutton = QPushButton("Selector")
 
# setting geometry of button
selecbutton.setGeometry(0, 30, 100, 30)

# adding action to a button
selecbutton.clicked.connect(self.clickme2)
 
    # action method
def clickme(self):
        
    selector.change_mode("angle")

def clickme2(self):
        
    selector.change_mode("selector")

app = pg.mkQApp("Angle Example")
window = pg.GraphicsLayoutWidget(show=True, size=(1000,800), border=True)


def main():
    #create selector object
    selector = classes.selector()

    # create pyqt5 app
    app = QApplication(sys.argv)
    
    # create the instance of our Window
    win = Window()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
