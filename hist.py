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
import main
from PyQt5.QtWidgets import*
from random import randint
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor, QFont

but_stat = True
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.i = 0
        self.e = 0
        self.step_hist = 0
        self.step_dft = 1
        self.step_dct = 2
        self.step_scl = 4
        self.step_grad = 3
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
        self.b = QInputDialog.getText(self, 'Введите число', 'Введите количество эталонных изображений:')

        self.b = int(self.b[0])
        self.result = main.Finder(self.b)

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

        self.vertical_layout.addWidget(self.button)
        self.vertical_layout.addWidget(self.button2)
        figure = self.canvas.figure
        figure.text(1, 1, "Надпись", color='red', ha="right", va="bottom")
        self.setCentralWidget(central_widget)
        self.button.clicked.connect(self.Stop)
        self.button2.clicked.connect(self.FinalResult)

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
        ax1 = self.canvas.figure.add_subplot(441)

        ax1.title.set_text('Тест')

        ax1.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        ####################################################################
        #Гистограмма
        ax2 = self.canvas.figure.add_subplot(443)
        ax2.title.set_text(f'Гистограмма {self.result[self.step_hist]}% ')
        img2 = cv.imread(et_path)
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv.calcHist([img], [i], None, [256], [0, 256])
            hist2 = cv.calcHist([img2], [i], None, [256], [0, 256])
            ax2.plot(hist, color='b')
            ax2.plot(hist2, color='r')
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
        ax3 = self.canvas.figure.add_subplot(444)
        ax3.title.set_text(f'Градиент {self.result[self.step_hist + 3]}')
        ax3.imshow(grad)

        ####################################################################
        # DCT
        ax4 = self.canvas.figure.add_subplot(445)
        imgcv1 = cv.split(img)[0]
        imf = np.float32(imgcv1) / 255.0  # float conversion/scale
        dct = cv.dct(imf)  # the dct
        imgcv1 = np.uint8(dct * 255.0)

        # Отображение результатов
        # ax4.title.set_text(f'DCT {self.result[self.step_hist + 2]}')
        ax4.set_xlabel(f'DCT {self.result[self.step_hist + 2]}')
        ax4.imshow(imgcv1)


        ax5 = self.canvas.figure.add_subplot(446)
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
        # ax5.title.set_text(f'DFT {self.result[self.i + 1]}')
        ax5.set_xlabel(f'DFT {self.result[self.step_hist + 1]}')
        ax5.imshow(magnitude)
        #########################################
        #Эталон
        ax6 = self.canvas.figure.add_subplot(442)
        ax6.title.set_text('Эталон')
        img = cv.imread(et_path)

        ax6.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        #########################################################################################################################
        # SCALE
        ax8 = self.canvas.figure.add_subplot(447)
        # ax8.title.set_text(f'SCALE {self.result[self.step_hist + 4]}')
        ax8.set_xlabel(f'SCALE {self.result[self.step_hist + 4]}')

        img = cv.imread(img_path)

        mg1_resized = cv.resize(img, (14, 12))

        # gray_img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # level = 3
        # coeffs2 = pywt.wavedec2(gray_img2, 'db2', mode='periodization', level=level)
        # threshold = 80
        # new_coeffs2 = list(coeffs2)
        # for i in range(1, level + 1):
        #     new_coeffs2[i] = tuple([np.where(np.abs(detail) < threshold, 0, detail) for detail in coeffs2[i]])
        # denoised_img1 = pywt.waverec2(new_coeffs2, 'db2', mode='periodization')




        # # Начальный размер изображения
        # height, width = img.shape[:2]
        # # Конечный размер изображения
        # new_height, new_width = int(height // 3), int(width // 3)
        #
        # # Установить уровень декомпозиции
        # level = 3
        # # Выполнить вейвлет-преобразование над изображением
        # coeffs = pywt.wavedec2(img, 'db2', mode='periodization', level=level)
        #
        # # Произвести изменение размера за счет удаления
        # # коэффициентов детализации вейвлет-преобразования на некоторых уровнях
        # new_coeffs = list(coeffs)
        # for i in range(1, level + 1):
        #     new_coeffs[i] = tuple([coeffs[i][j][:new_height, :new_width] for j in range(len(coeffs[i]))])
        # # Выполнить обратное вейвлет-преобразование
        # resized_img = pywt.waverec2(new_coeffs, 'db2', mode='periodization')
        ax8.imshow(mg1_resized)
        #########################################################################################################################
        ax7 = self.canvas.figure.add_subplot(313)
        ax7.title.set_text('График определения лица по Гистограмме')

        self.x_data.append(self.i)
        self.y_data.append(self.result[self.step_hist])
        self.y_data_dft.append(self.result[self.step_dft])
        self.y_data_grad.append(self.result[self.step_grad])
        self.y_data_dct.append(self.result[self.step_dct])
        self.y_data_scl.append(self.result[self.step_scl])
        ax7.axes.clear()
        ax7.axes.plot(self.x_data, self.y_data,color='r', label='Hist')
        ax7.axes.plot(self.x_data, self.y_data_dft,color='b', label='DFT')
        ax7.axes.plot(self.x_data, self.y_data_grad, color='g', label='Grad')
        ax7.axes.plot(self.x_data, self.y_data_dct, color='y', label='DCT')
        ax7.axes.plot(self.x_data, self.y_data_scl, color='c', label='SCL')
        ax7.legend()
        ax7.set_xlabel('Номер тестового изображения')
        ax7.set_ylabel('Проценты')

        # ax9 = self.canvas.figure.add_subplot(614)
        # ax9.title.set_text('График сходимости по DFT')
        # self.x_data_dft.append(self.i)
        # self.y_data_dft.append(self.result[self.step_dft])
        # ax9.axes.clear()
        # ax9.axes.plot(self.x_data_dft, self.y_data_dft)
        #
        # ax9.set_xlabel('Номер тестового изображения')
        # ax9.set_ylabel('Проценты')
        # ax1.clear()
        # ax2.clear()
        # ax3.clear()
        # ax4.clear()
        # ax5.clear()
        # ax6.clear()
        self.avg.append((self.result[self.step_hist] + self.result[self.step_dft] + self.result[self.step_grad] + self.result[self.step_dct] + self.result[self.step_scl])/5)

        self.lable.setText(f'текущая точность распознавания лица: {round(self.avg[self.i],1)} %')




        # ax7.plot(self.i,self.result[self.i])
        self.canvas.draw()
        self.i = self.i + 1
        self.step_dft =self.step_dft + 5
        self.step_hist = self.step_hist + 5
        self.step_grad = self.step_grad + 5
        self.step_dct= self.step_dct + 5
        self.step_scl= self.step_scl + 5
        if (self.i % (10 - self.b) == 0):
            self.e = self.e + 1

    def FinalResult(self):
        self.lable.setText(f'Средняя точность системы распознования лиц: {round(sum(self.result)/len(self.result), 1)} %')
        self.timer.stop()
        self.but_stat = False

        print(len(self.result))
        data = {}
        y_data = []
        y_data_dft = []
        y_data_grad = []
        y_data_dct = []
        y_data_scl = []
        i_test = []
        e_etalon = []
        i = 0
        e = 0
        step_dft = 1
        step_hist = 0
        step_grad = 3
        step_dct = 2
        step_scl = 4

        for i in range(len(self.result)//5 - 5):
            y_data.append(self.result[step_hist])
            y_data_dft.append(self.result[step_dft])
            y_data_grad.append(self.result[step_grad])
            y_data_dct.append(self.result[step_dct])
            y_data_scl.append(self.result[step_scl])
            i_test.append(i)
            e_etalon.append(e)
            step_dft = step_dft + 5
            step_hist = step_hist + 5
            step_grad = step_grad + 5
            step_dct = step_dct + 5
            step_scl = step_scl + 5

            if( i % 5 == 0):
                i = i + 1
            if (i % (9 - self.b) == 0):
                e = e + 1

        data= {
            'Гистограмма': y_data,
            'DFT': y_data_dft,
            'DCT': y_data_dct,
            'Градиент': y_data_grad,
            'Scale': y_data_scl,
            'Номер тестового изображения': i_test,
            'Номер Эталонного изображения': e_etalon,
            'Средняя точность системы разпознования лиц': round(sum(self.result)/len(self.result),1)
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








#
app = QtWidgets.QApplication([])
window = MainWindow()
window.show()
app.exec_()

