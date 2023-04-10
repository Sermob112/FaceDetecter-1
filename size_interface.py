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
class Window_Scale(QtWidgets.QMainWindow):

    def __init__(self, tests,dct_result ):
        super().__init__()
        self.i = 0
        self.e = 0
        self.k = 0
        self.step_dct = 2
        self.tests = tests
        self.dct_result = dct_result
        self.but_stat = True
        self.b = 0

        self.x_data = []
        self.y_data = []
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
        for i in range(len(self.dct_result)):
            self.x_data.append(i)


        figure = self.canvas.figure
        figure.text(1, 1, "Надпись", color='red', ha="right", va="bottom")
        self.setCentralWidget(central_widget)
        self.button.clicked.connect(self.Stop)

    def get_all(self,tests, dct_result):
        self.tests = tests
        self.dct_result = dct_result
    def load_image(self):
        self.canvas.figure.clear()
        # img_path = self.tests[self.i]
        # et_path = standards[self.e]
        # img = cv.imread(img_path)
        # ax1 = self.canvas.figure.add_subplot(121)
        #
        # ax1.title.set_text('Тест')
        #
        # ax1.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        #
        # ax_e = self.canvas.figure.add_subplot(122)



        ax2 = self.canvas.figure.add_subplot(111)
        ax2.title.set_text('График определения лица по SCALE')
        # self.x_data.append(self.i)
        self.y_data.append(self.dct_result[self.i])
        ax2.axes.clear()
        ax2.axes.plot(self.x_data, self.dct_result, color='r', label='SCALE')
        ax2.legend()
        ax2.set_xlabel('Номер теста')
        ax2.set_ylabel('Проценты')
        self.avg.append(self.dct_result[self.i])
        result = (sum(self.dct_result) / len(self.dct_result))
        self.lable.setText(f'точность распознавания лица методом SCALE: {round(result, 5)} %')
        self.canvas.draw()
        self.i = self.i + 1

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