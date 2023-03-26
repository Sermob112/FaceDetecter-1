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
        self.i = 0
        self.e = 0
        self.step = 0
        self.x_data = []
        self.y_data = []
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
        if (self.e == 3):
            self.timer.stop()

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
        ax1 = self.canvas.figure.add_subplot(341)

        ax1.title.set_text('Тест')

        ax1.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))


        #Гистограмма
        ax2 = self.canvas.figure.add_subplot(343)
        ax2.title.set_text(f'Гистограмма {self.result[self.step]}% ')

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
        ax3 = self.canvas.figure.add_subplot(344)
        ax3.title.set_text(f'Градиент {self.result[self.step + 3]}')
        ax3.imshow(grad)

        ####################################################################
        # DCT
        ax4 = self.canvas.figure.add_subplot(345)
        imgcv1 = cv.split(img)[0]
        imf = np.float32(imgcv1) / 255.0  # float conversion/scale
        dct = cv.dct(imf)  # the dct
        imgcv1 = np.uint8(dct * 255.0)

        # Отображение результатов
        ax4.title.set_text(f'DCT {self.result[self.step + 2]}')
        ax4.imshow(imgcv1)


        ax5 = self.canvas.figure.add_subplot(346)
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
        ax5.title.set_text(f'DFT {self.result[self.i + 1]}')
        ax5.imshow(magnitude)
        #########################################
        #Эталон
        ax6 = self.canvas.figure.add_subplot(342)
        ax6.title.set_text('Эталон')
        img = cv.imread(et_path)
        ax6.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        #########################################
        # SCALE
        ax8 = self.canvas.figure.add_subplot(347)
        ax8.title.set_text(f'SCALE {self.result[self.step + 4]}')
        scale_up_x = 1.2
        scale_up_y = 1.2
        img = cv.imread(img_path)
        scale_down = 0.2
        scaled_f_down = cv.resize(img, None, fx= scale_down, fy= scale_down, interpolation= cv.INTER_LINEAR)
        ax8.imshow(cv.cvtColor(scaled_f_down, cv.COLOR_BGR2RGB))

        ax7 = self.canvas.figure.add_subplot(414)
        ax7.title.set_text('График сходимости по Гистограмме')
        self.x_data.append(self.i)
        self.y_data.append(self.result[self.step])
        ax7.axes.clear()
        ax7.axes.plot(self.x_data, self.y_data)

        ax7.set_xlabel('Номер теста')
        ax7.set_ylabel('Проценты')

        # ax1.clear()
        # ax2.clear()
        # ax3.clear()
        # ax4.clear()
        # ax5.clear()
        # ax6.clear()

        # ax7.plot(self.i,self.result[self.i])

        self.i = self.i + 1
        self.step = self.step + 5
        if (self.i % 9 == 0):
            self.e = self.e + 1
        self.canvas.draw()







#
app = QtWidgets.QApplication([])
window = MainWindow()
window.show()
app.exec_()
