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
from Dct_interface import Window_DCT
but_stat = True
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.i = 0
        self.e = 0
        self.all_test = 0
        self.col = 0
        self.k = 0
        self.step_hist = 0
        self.step_dft = 1
        self.step_dct = 2
        self.step_scl = 4
        self.step_grad = 3
        self.but_stat = True
        self.b = 0
        self.d = 0
        self.x_data = []
        self.y_data = []
        self.y_data_dft = []
        self.y_data_grad = []
        self.y_data_dct = []
        self.y_data_scl = []

        self.avg = []
        self.final_avg = []
        self.b = QInputDialog.getInt(self, 'Введите число', 'Введите количество эталонных изображений внутри одного класса:')
        self.d = QInputDialog.getInt(self, 'Введите число', 'Введите количество классов:')
        self.b = self.b[0]
        self.d = self.d[0]


        final_validation(self.d, self.b)

        # self.hist_result = hist_result
        # self.dct_result = dct_result
        # self.dft_result = dft_result
        # self.grad_result = []
        # self.scale_result = []

        self.setWindowTitle("Процент точности системы")
        central_widget = QtWidgets.QWidget(self)
        self.vertical_layout = QtWidgets.QVBoxLayout(central_widget)
        self.canvas = FigureCanvas(Figure(figsize=(15, 15)))
        self.vertical_layout.addWidget(self.canvas)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.load_image)
        self.timer.start(100)
        self.lable = QtWidgets.QLabel(f'0')
        self.lable.setAlignment(Qt.AlignHCenter)
        self.lable.setFont(QFont('Times', 16))
        self.vertical_layout.addWidget(self.lable)
        self.button = QtWidgets.QPushButton("Остановить")
        self.button2 = QtWidgets.QPushButton("Результаты")

        self.button_hist = QtWidgets.QPushButton("Гистограмма")
        self.button_DCT = QtWidgets.QPushButton("DCT")
        self.button_DFT = QtWidgets.QPushButton("DFT")
        self.button_Sclae = QtWidgets.QPushButton("Scale")
        self.button_grad = QtWidgets.QPushButton("Градиент")

        self.vertical_layout.addWidget(self.button)
        self.vertical_layout.addWidget(self.button2)
        self.vertical_layout.addWidget(self.button_hist)
        self.vertical_layout.addWidget(self.button_DCT)
        self.vertical_layout.addWidget(self.button_DFT)
        self.vertical_layout.addWidget(self.button_Sclae)
        self.vertical_layout.addWidget(self.button_grad)

        # self.window_dct = Window_DCT()
        # self.button_DCT.clicked.connect(self.open_Dct)

        figure = self.canvas.figure
        figure.text(1, 1, "Надпись", color='red', ha="right", va="bottom")
        self.setCentralWidget(central_widget)
        self.button.clicked.connect(self.Stop)
        self.button2.clicked.connect(self.FinalResult)

    # def open_Dct(self):
    #     self.window_dct.show()
    def load_image(self):
        for l in range(self.b):
            try:
                self.canvas.figure.clear()
                img_path = tests[self.i + self.k]
                et_path = standards[self.e]

                img = cv.imread(img_path)
                ####################################################################
                #Оригинал
                #строки, стобец, место в столбце
                ax1 = self.canvas.figure.add_subplot(441)

                ax1.title.set_text('Тест')

                ax1.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

                # ####################################################################

                #Эталон
                ax6 = self.canvas.figure.add_subplot(442)
                ax6.title.set_text('Эталон')
                img_et = cv.imread(et_path)

                ax6.imshow(cv.cvtColor(img_et, cv.COLOR_BGR2RGB))

                # #########################################################################################################################
                # #Гистограмма
                ax2 = self.canvas.figure.add_subplot(443)
                ax2.title.set_text(f'Гистограмма {hist_result[self.i]}% ')
                img2 = cv.imread(et_path)
                colors = ('b', 'g', 'r')
                for i, col in enumerate(colors):
                    hist = cv.calcHist([img], [i], None, [256], [0, 256])
                    hist2 = cv.calcHist([img2], [i], None, [256], [0, 256])
                    ax2.plot(hist, color='b')
                    ax2.plot(hist2, color='r')
                    ax2.set_xlim([0, 256])
                #
                # ####################################################################
                # # Градиент
                sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
                sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

                # Вычисление абсолютного значения градиента
                abs_sobelx = cv.convertScaleAbs(sobelx)
                abs_sobely = cv.convertScaleAbs(sobely)

                # Объединение градиентов по горизонтали и вертикали
                grad = cv.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

                # Отображение изображения с градиентом
                ax3 = self.canvas.figure.add_subplot(444)
                ax3.title.set_text(f'Градиент {grad_result[self.i]}')
                ax3.imshow(grad)
                #
                # ####################################################################
                # # DCT
                ax4 = self.canvas.figure.add_subplot(445)
                imgcv1 = cv.split(img)[0]
                imf = np.float32(imgcv1) / 255.0  # float conversion/scale
                dct = cv.dct(imf)  # the dct
                imgcv1 = np.uint8(dct * 255.0)
                #
                # # Отображение результатов
                # ax4.title.set_text(f'DCT {self.result[self.step_hist + 2]}')
                ax4.set_xlabel(f'DCT {dct_result[self.i]}')
                ax4.imshow(imgcv1)
                #
                #
                ####DFT
                ax5 = self.canvas.figure.add_subplot(446)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                fourier = cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)
                fourier_shift = np.fft.fftshift(fourier)
                magnitude = 20 * np.log(cv.magnitude(fourier_shift[:, :, 0], fourier_shift[:, :, 1]))
                magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
                ax5.set_xlabel(f'DFT {dft_result[self.i]}')
                ax5.imshow(magnitude)
                # #########################################
                #
                #
                # #########################################################################################################################
                # # SCALE
                ax8 = self.canvas.figure.add_subplot(447)
                ax8.set_xlabel(f'SCALE {scale_result[self.i]}')

                img = cv.imread(img_path)

                mg1_resized = cv.resize(img, (14, 12))

                ax8.imshow(mg1_resized)
                # #########################################################################################################################
                ax7 = self.canvas.figure.add_subplot(313)
                ax7.title.set_text('График определения лица по Гистограмме')

                self.x_data.append(self.all_test)
                self.y_data.append(hist_result[self.i])
                self.y_data_dft.append(dft_result[self.i])
                self.y_data_grad.append(grad_result[self.i])
                self.y_data_dct.append(dct_result[self.i])
                self.y_data_scl.append(scale_result[self.i])
                ax7.axes.clear()
                ax7.axes.plot(self.x_data, self.y_data,color='r', label='Hist')
                ax7.axes.plot(self.x_data, self.y_data_dft,color='b', label='DFT')
                ax7.axes.plot(self.x_data, self.y_data_grad, color='g', label='Grad')
                ax7.axes.plot(self.x_data, self.y_data_dct, color='y', label='DCT')
                ax7.axes.plot(self.x_data, self.y_data_scl, color='c', label='SCL')
                ax7.legend()
                ax7.set_xlabel('Номер тестового изображения')
                ax7.set_ylabel('Проценты')
                #
                self.avg.append((hist_result[self.i] + dft_result[self.i] + grad_result[self.i] + dct_result[self.i] + scale_result[self.i])/5)
                #
                self.lable.setText(f'текущая точность распознавания лица: {round(self.avg[self.i],1)} %')
                #
                # # ax7.plot(self.i,self.result[self.i])
                self.canvas.draw()
                self.i = self.i + 1
                self.all_test = self.all_test + 1
                if (self.i % (10 - self.b) == 0):
                    self.e = self.e + 1
                    self.i = 0
                    self.col = self.col + 1
                    if (self.col % self.b == 0):
                        self.col = 0
                        self.k =self.k + 10 - self.b

            except Exception:
                self.but_stat = False

    def FinalResult(self):

        self.timer.stop()
        self.but_stat = False


        data = {}
        i_test = []
        e_etalon = []
        avg = []
        avg = (hist_result + dft_result + dct_result + grad_result + scale_result)
        result = round(sum(avg)/len(avg),1)
        self.lable.setText(f'Средняя точность системы распознования лиц: {result} %')
        h= 0
        e = 0
        k = 0
        g = 0
        f = 0
        etl_files = []
        test_files = []

        nb = len(standards)
        kt = len(hist_result)
        khlf= len(tests)
        for c in range(1, len(hist_result) + 1):
            etl_files.append(standards[e])
            e_etalon.append(e)
            i_test.append(c)
            try:
                test_files.append(tests[g + k])
            except Exception:
                test_files.append(tests[k])
            # if (c % (10 - self.b) == 0):
            #     # e = e + 1
            #     k = k + 10 - self.b

            if (c % (10 - self.b) == 0):
                h = h + 1
                g  = 0
                if (h == self.b):
                    e = e + 1
                    k = k + (10 - self.b)
                    h = 0
            g = g + 1


            # for l in range(self.b):
            #     for h in range(10 - self.b):
            #         i_test.append(h + k)
            # k = k + 10 - self.b

        #
        data= {
            'Гистограмма': hist_result,
            'DFT': dft_result,
            'DCT': dct_result,
            'Градиент': grad_result,
            'Scale': scale_result,
            'Номер тестового изображения': i_test,
            'Номер Эталонного изображения': e_etalon,
            'Средняя точность системы разпознования лиц': result,
            'Файл тестового изображения': test_files,
            'файл эталонного изображения':etl_files
        }
        df = pd.DataFrame(data)
        df.to_excel(f'Результаты работы с {self.b} эталонами.xlsx', index=False)

    def Stop(self):
        if(self.but_stat == True):
            self.timer.stop()
            self.but_stat = False
        else:
            self.timer.start(100)
            self.but_stat = True

# app = QtWidgets.QApplication([])
# window = MainWindow()
# window.show()
# app.exec_()