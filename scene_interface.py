import cv2 as cv
import numpy as np
from PyQt5.QtCore import QTimer, Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor, QFont
from scene_decter import *
but_stat = True
class Window_DCT(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.i = 0
        self.e = 1
        self.but_stat = True
        self.b = 0
        select_files(500)
        self.data_proc = find_scene()

        self.x_data = []
        self.y_data = []
        self.avg = []
        self.final_avg = []
        self.setWindowTitle("Определение момента изменения сцены")
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
        self.vertical_layout.addWidget(self.button)



        figure = self.canvas.figure
        figure.text(1, 1, "Надпись", color='red', ha="right", va="bottom")
        self.setCentralWidget(central_widget)
        self.button.clicked.connect(self.Stop)


    def load_image(self):
        self.canvas.figure.clear()
        try:
            img_path = standards[self.i]
            et_path = standards[self.e]

            img = cv.imread(img_path)
            ####################################################################
            # Оригинал
            # строки, стобец, место в столбце
            ax1 = self.canvas.figure.add_subplot(221)

            ax1.title.set_text('Тест')

            ax1.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

            # ####################################################################
            ax2 = self.canvas.figure.add_subplot(222)
            ax2.title.set_text('Эталон')
            img_et = cv.imread(et_path)

            ax2.imshow(cv.cvtColor(img_et, cv.COLOR_BGR2RGB))
            #Гистограмма
            ax3 = self.canvas.figure.add_subplot(212)
            ax3.title.set_text(f'Гистограмма яркости% ')
            img2 = cv.imread(et_path)
            # colors = ('b', 'g', 'r')
            # for i, col in enumerate(colors):
            hist = cv.calcHist([img], [0], None, [256], [0, 256])
            hist2 = cv.calcHist([img2], [0], None, [256], [0, 256])
            ax3.plot(hist, color='b',label='Текущий кадр')
            ax3.plot(hist2, color='r',label='Следующий кадр')
            ax3.legend()
            ax3.set_xlim([0, 256])
            # ax3.set_ylabel('Проценты')



            # ax2 = self.canvas.figure.add_subplot(111)
            # ax2.title.set_text('График определения лица по DCT')
            # self.x_data.append(self.i)
            # self.y_data.append(self.dct_result[self.i])
            # ax2.axes.clear()
            # ax2.axes.plot(self.x_data, self.y_data, color='y', label='DCT')
            # ax2.legend()
            # ax2.set_xlabel('Номер тестового изображения')
            # ax2.set_ylabel('Проценты')
            # self.avg.append(self.dct_result[self.i])
            # self.final_avg.append(sum(self.avg)/len(self.avg))
            self.lable.setText(f'текущая разность гистограммы яркости в % : {self.data_proc[self.i]} %')
            self.canvas.draw()
            self.i = self.i + 1
            self.e = self.e + 1
        except Exception:
            self.e = len(standards)

    def Stop(self):
        if(self.but_stat == True):
            self.timer.stop()
            self.but_stat = False
        else:
            self.timer.start(100)
            self.but_stat = True

app = QtWidgets.QApplication([])
window = Window_DCT()
window.show()
app.exec_()