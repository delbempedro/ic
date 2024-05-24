import classes
import pyqtgraph as pg
import sys

from pyqtgraph.Qt.QtWidgets import *
from pyqtgraph.Qt.QtGui import *
from pyqtgraph.Qt.QtCore import *

pg.setConfigOptions(imageAxisOrder='row-major')

#create selector object
selector = classes.selector()

## create GUI
app = pg.mkQApp("ROI Examples")

# create the instance of our Window
win = pg.GraphicsLayoutWidget(show=True, size=(1000,800), border=True)
win.setWindowTitle('pyqtgraph example: ROI Examples')

# creating a push button
anglebutton = pg.PathButton("Angle")
 
# setting geometry of button
anglebutton.setGeometry(0, 0, 100, 30)
 
# adding action to a button
"""anglebutton.clicked.connect(self.clickme)"""

# creating a push button
selecbutton = pg.PathButton("Selector")
 
# setting geometry of button
selecbutton.setGeometry(0, 30, 100, 30)

"""# adding action to a button
selecbutton.clicked.connect(self.clickme2)
 
    # action method
def clickme(self):
        
    selector.change_mode("angle")

def clickme2(self):
        
    selector.change_mode("selector")    """


sys.exit(app.exec_())

if __name__ == "__main__":
    pg.exec()
