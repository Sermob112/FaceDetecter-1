import sys
import cv2 as cv
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from random import randint

import main


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.i = 1
    def init_ui(self):

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.setCentralWidget(self.canvas)

        self.result = main.Finder()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_figure)
        self.timer.start(1000)


    def update_figure(self):
        self.figure.clear()
        img = cv.imread(main.test[self.i])

        # Преобразование цвета изображения из BGR в RGB
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Добавить изображение в FigureCanvas
        ax = self.figure.add_subplot(1, 1, 1)
        ax.imshow(img)
        self.i = self.i + 1
        self.canvas.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())