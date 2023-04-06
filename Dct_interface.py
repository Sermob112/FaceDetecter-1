import cv2 as cv
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
# from mtcnn.mtcnn import MTCNN
import math
import pandas as pd
import os
import sys

import pywt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from methods import *
from PyQt5.QtWidgets import*
from random import randint
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor, QFont

but_stat = True
class Window_DCT(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.i = 0
        self.e = 0
        self.k = 0
        self.step_dct = 2

        self.but_stat = True
        self.b = 0

        self.x_data = []
        self.y_data = []
        self.y_data_dft = []
        self.y_data_grad = []
        self.y_data_dct = []
        self.y_data_scl = []

        self.avg = []
        self.final_avg = []
        self.setWindowTitle("Процент точности DCT")
        central_widget = QtWidgets.QWidget(self)
        self.vertical_layout = QtWidgets.QVBoxLayout(central_widget)
        self.canvas = FigureCanvas(Figure(figsize=(15, 15)))
        self.vertical_layout.addWidget(self.canvas)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.load_image)
        self.timer.start(1000)
        self.lable = QtWidgets.QLabel(f'0')
        self.lable.setAlignment(Qt.AlignHCenter)
        self.lable.setFont(QFont('Times', 16))
        self.vertical_layout.addWidget(self.lable)
        self.button = QtWidgets.QPushButton("Остановить")
        self.vertical_layout.addWidget(self.button)



        figure = self.canvas.figure
        figure.text(1, 1, "Надпись", color='red', ha="right", va="bottom")
        self.setCentralWidget(central_widget)
        self.button.clicked.connect(self.Stop)
        # self.button2.clicked.connect(self.FinalResult)
    def get_all(self, test, result):
        self.test = test
        self.result = result
    def load_image(self):
        self.canvas.figure.clear()
        img_path = self.test[self.i]

        img = cv.imread(img_path)
        ax1 = self.canvas.figure.add_subplot(121)

        ax1.title.set_text('Тест')

        ax1.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        ax2 = self.canvas.figure.add_subplot(122)
        ax2.title.set_text('График определения лица по DCT')
        self.x_data.append(self.i)
        self.y_data.append(self.result[self.step_dct])
        ax2.axes.clear()
        ax2.axes.plot(self.x_data, self.y_data, color='y', label='DCT')
        ax2.legend()
        ax2.set_xlabel('Номер тестового изображения')
        ax2.set_ylabel('Проценты')

        self.canvas.draw()
        self.i = self.i + 1
        self.step_dct = self.step_dct + 5
    def Stop(self):
        if(self.but_stat == True):
            self.timer.stop()
            self.but_stat = False
        else:
            self.timer.start(100)
            self.but_stat = True

# app = QtWidgets.QApplication([])
# window = Window_DCT()
# window.show()
# app.exec_()