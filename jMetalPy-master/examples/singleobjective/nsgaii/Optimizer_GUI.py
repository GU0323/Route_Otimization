import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_ui = uic.loadUiType("optimizer.ui")[0]

class MyWindow(QMainWindow, form_ui):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)


if __name__=="__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()