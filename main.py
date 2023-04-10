from PyQt5 import QtWidgets, uic,QtCore,QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow
# from All_Interface import MainWindow
from All_Interface_Test import MainWindow
import sys

from methods import Finder


def application():
    app = QtWidgets.QApplication([])
    window = MainWindow()

    window.show()
    app.exec_()

application()
