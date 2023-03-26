import cv2 as cv
import numpy as np
from PyQt5.QtCore import QTimer
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mtcnn.mtcnn import MTCNN
import math
import pandas as pd
import os
import sys

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import main
from PyQt5.QtWidgets import*
from random import randint
from PyQt5 import QtCore, QtGui, QtWidgets


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.i = 1
        self.e = 1
        self.result = main.Finder()
        # set the title of the main window
        self.setWindowTitle("Image Histogram")

        # create a central widget to hold the canvas and the button
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)

        # create a vertical box layout to hold the canvas and the button
        vertical_layout = QtWidgets.QVBoxLayout(central_widget)

        # create a canvas to display the image
        self.canvas = FigureCanvas(Figure(figsize=(10, 5)))
        vertical_layout.addWidget(self.canvas)



        self.timer = QTimer(self)
        self.timer.timeout.connect(self.load_image)
        self.timer.start(100)  # обновление каждую секунду

        # create a button to load the image
        self.button = QtWidgets.QPushButton("Показать результат")
        vertical_layout.addWidget(self.button)

        # connect the button to the function that loads the image
        self.button.clicked.connect(self.load_image)

    # def update_figure(self):

        # ax = self.canvas.figure.add_subplot()
        # # Очистите текущий график и нарисуйте новый
        # ax.clear()
        # ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        # ax.set_title("Updated Plot")
        # # Обновите FigureCanvas
        # self.canvas.draw()
        # self.canvas.clear()
    def load_image(self):
        self.canvas.figure.clear()
        img_path = main.test[self.i]
        et_path = main.etalon[self.e]
        img = cv.imread(img_path)
        ####################################################################
        #Оригинал
        # create a subplot for the image
        #строки, стобец, место в столбце
        ax1 = self.canvas.figure.add_subplot(331)

        ax1.title.set_text('Тест')

        ax1.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))


        #Гистограмма
        ax2 = self.canvas.figure.add_subplot(333)
        ax2.title.set_text('Гистограмма')

        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv.calcHist([img], [i], None, [256], [0, 256])
            ax2.plot(hist, color=col)
            ax2.set_xlim([0, 256])

        ####################################################################
        # Градиент
        sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
        sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

        # Вычисление абсолютного значения градиента
        abs_sobelx = cv.convertScaleAbs(sobelx)
        abs_sobely = cv.convertScaleAbs(sobely)

        # Объединение градиентов по горизонтали и вертикали
        grad = cv.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

        # Отображение изображения с градиентом
        ax3 = self.canvas.figure.add_subplot(334)
        ax3.title.set_text('Градиент')
        ax3.imshow(grad)

        ####################################################################
        # DCT
        ax4 = self.canvas.figure.add_subplot(335)
        imgcv1 = cv.split(img)[0]
        imf = np.float32(imgcv1) / 255.0  # float conversion/scale
        dct = cv.dct(imf)  # the dct
        imgcv1 = np.uint8(dct * 255.0)

        # Отображение результатов
        ax4.title.set_text('DCT')
        ax4.imshow(imgcv1)


        ax5 = self.canvas.figure.add_subplot(336)
        ####DFT

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Compute the discrete Fourier Transform of the image
        fourier = cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)

        # Shift the zero-frequency component to the center of the spectrum
        fourier_shift = np.fft.fftshift(fourier)

        # calculate the magnitude of the Fourier Transform
        magnitude = 20 * np.log(cv.magnitude(fourier_shift[:, :, 0], fourier_shift[:, :, 1]))

        # Scale the magnitude for display
        magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)

        # Display the magnitude of the Fourier Transform
        ax5.title.set_text('DFT')
        ax5.imshow(magnitude)


        ax6 = self.canvas.figure.add_subplot(332)
        ax6.title.set_text('Эталон')



        img = cv.imread(et_path)
        ax6.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))


        ax7 = self.canvas.figure.add_subplot(414)
        ax7.set_xlabel('test')
        ax7.set_ylabel('test')
        # ax1.clear()
        # ax2.clear()
        # ax3.clear()
        # ax4.clear()
        # ax5.clear()
        # ax6.clear()
        j = []
        for i in range(len(self.result)):
            j.append(i)
        ax7.plot(j,self.result)
        self.i = self.i + 1
        if (self.i % 10 == 0):
            self.e = self.e + 1
        self.canvas.draw()






#
app = QtWidgets.QApplication([])
window = MainWindow()
window.show()
app.exec_()
