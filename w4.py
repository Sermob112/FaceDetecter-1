# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'w4.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import cv2 as cv
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer
from methods import *
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.i = 0
        self.e = 0
        self.step_hist = 0
        self.step_dft = 1
        self.step_dct = 2
        self.step_scl = 4
        self.step_grad = 3

        self.x_data = []
        self.y_data = []

        self.y_data_dft = []
        self.y_data_grad = []
        self.y_data_dct = []
        self.y_data_scl = []

        self.result = Finder()
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 520)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.vertical_layout = QtWidgets.QVBoxLayout(self.centralwidget)

        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.vertical_layout.addWidget(self.canvas)
        self.timer = QTimer()
        self.timer.timeout.connect(self.load_image)
        self.timer.start(100)


        self.widget.setGeometry(QtCore.QRect(0, 50, 791, 351))
        self.widget.setObjectName("widget")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(570, 260, 201, 61))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(40, 420, 101, 41))
        self.pushButton.setObjectName("pushButton")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(600, 420, 161, 41))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton.setText(_translate("MainWindow", "PushButton"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))

    def load_image(self):
        self.canvas.figure.clear()
        img_path = test[self.i]
        et_path = etalon[self.e]

        img = cv.imread(img_path)
        ####################################################################
        # Оригинал
        # create a subplot for the image
        # строки, стобец, место в столбце
        ax1 = self.canvas.figure.add_subplot(341)

        ax1.title.set_text('Тест')

        ax1.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        # Гистограмма
        ax2 = self.canvas.figure.add_subplot(343)
        ax2.title.set_text(f'Гистограмма {self.result[self.step_hist]}% ')

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
        ax3.title.set_text(f'Градиент {self.result[self.step_hist + 3]}')
        ax3.imshow(grad)

        ####################################################################
        # DCT
        ax4 = self.canvas.figure.add_subplot(345)
        imgcv1 = cv.split(img)[0]
        imf = np.float32(imgcv1) / 255.0  # float conversion/scale
        dct = cv.dct(imf)  # the dct
        imgcv1 = np.uint8(dct * 255.0)

        # Отображение результатов
        ax4.title.set_text(f'DCT {self.result[self.step_hist + 2]}')
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
        # Эталон
        ax6 = self.canvas.figure.add_subplot(342)
        ax6.title.set_text('Эталон')
        img = cv.imread(et_path)
        ax6.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        #########################################
        # SCALE
        ax8 = self.canvas.figure.add_subplot(347)
        ax8.title.set_text(f'SCALE {self.result[self.step_hist + 4]}')
        scale_up_x = 1.2
        scale_up_y = 1.2
        img = cv.imread(img_path)
        scale_down = 0.2
        scaled_f_down = cv.resize(img, None, fx=scale_down, fy=scale_down, interpolation=cv.INTER_LINEAR)
        ax8.imshow(cv.cvtColor(scaled_f_down, cv.COLOR_BGR2RGB))

        ax7 = self.canvas.figure.add_subplot(414)
        ax7.title.set_text('График определения лица по Гистограмме')

        self.x_data.append(self.i)
        self.y_data.append(self.result[self.step_hist])
        self.y_data_dft.append(self.result[self.step_dft])
        self.y_data_grad.append(self.result[self.step_grad])
        self.y_data_dct.append(self.result[self.step_dct])
        self.y_data_scl.append(self.result[self.step_scl])
        ax7.axes.clear()
        ax7.axes.plot(self.x_data, self.y_data, color='r', label='sin')
        # ax7.axes.plot(self.x_data, self.y_data_dft,color='b', label='sin')
        ax7.axes.plot(self.x_data, self.y_data_grad, color='g', label='sin')
        ax7.axes.plot(self.x_data, self.y_data_dct, color='y', label='sin')
        ax7.axes.plot(self.x_data, self.y_data_scl, color='c', label='sin')
        ax7.set_xlabel('Номер тестового изображения')
        ax7.set_ylabel('Проценты')
        self.canvas.draw()
        self.i = self.i + 1
        self.step_dft = self.step_dft + 5
        self.step_hist = self.step_hist + 5
        self.step_grad = self.step_grad + 5
        self.step_dct = self.step_dct + 5
        self.step_scl = self.step_scl + 5
        if (self.i % 9 == 0):
            self.e = self.e + 1

class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

def application():
    app = QApplication(sys.argv)
    window = mywindow()

    window.show()
    sys.exit(app.exec_())
if __name__ == '__main__':
    application()