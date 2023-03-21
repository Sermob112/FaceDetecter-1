import cv2 as cv
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mtcnn.mtcnn import MTCNN
import math
import pandas as pd
import os

from PyQt5.QtGui import QImage
from PyQt5 import QtCore, QtGui, QtWidgets
dir_path = 's12/'
dir_path2 = 'pic/'
res = []
res2 = []
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        res.append(dir_path + path)

for path2 in os.listdir(dir_path2):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path2, path2)):
        res2.append(dir_path2 + path2)



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(970, 841)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(550, 590, 161, 21))
        self.comboBox.setObjectName("comboBox")
        self.Template_but = QtWidgets.QPushButton(self.centralwidget)
        self.Template_but.setGeometry(QtCore.QRect(30, 650, 281, 51))
        self.Template_but.setStyleSheet("font: 16pt \"Lucida Console\";\n"
"background-color: rgb(0, 170, 0);")
        self.Template_but.setObjectName("Template_but")
        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setGeometry(QtCore.QRect(170, 590, 151, 21))
        self.comboBox_2.setObjectName("comboBox_2")
        self.result = QtWidgets.QLabel(self.centralwidget)
        self.result.setGeometry(QtCore.QRect(40, 280, 741, 281))
        self.result.setObjectName("result")
        self.tmplate = QtWidgets.QLabel(self.centralwidget)
        self.tmplate.setGeometry(QtCore.QRect(10, 40, 241, 221))
        self.tmplate.setObjectName("tmplate")
        self.pictures = QtWidgets.QLabel(self.centralwidget)
        self.pictures.setGeometry(QtCore.QRect(340, 30, 361, 221))
        self.pictures.setObjectName("pictures")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(300, 230, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("\n"
"background-color: rgb(85, 170, 255);")
        self.label_4.setFrameShape(QtWidgets.QFrame.Box)
        self.label_4.setLineWidth(-1)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(20, 10, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("\n"
"background-color: rgb(85, 170, 255);")
        self.label_5.setFrameShape(QtWidgets.QFrame.Box)
        self.label_5.setLineWidth(-1)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(350, 10, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("\n"
"background-color: rgb(85, 170, 255);")
        self.label_6.setFrameShape(QtWidgets.QFrame.Box)
        self.label_6.setLineWidth(-1)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(160, 550, 181, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("\n"
"background-color: rgb(85, 170, 255);")
        self.label_7.setFrameShape(QtWidgets.QFrame.Box)
        self.label_7.setLineWidth(-1)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(550, 550, 181, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("\n"
"background-color: rgb(85, 170, 255);")
        self.label_8.setFrameShape(QtWidgets.QFrame.Box)
        self.label_8.setLineWidth(-1)
        self.label_8.setObjectName("label_8")
        self.Viola_but = QtWidgets.QPushButton(self.centralwidget)
        self.Viola_but.setGeometry(QtCore.QRect(340, 650, 281, 51))
        self.Viola_but.setStyleSheet("font: 16pt \"Lucida Console\";\n"
"background-color: rgb(0, 170, 0);")
        self.Viola_but.setObjectName("Viola_but")
        self.Sim_but = QtWidgets.QPushButton(self.centralwidget)
        self.Sim_but.setGeometry(QtCore.QRect(670, 650, 281, 51))
        self.Sim_but.setStyleSheet("font: 16pt \"Lucida Console\";\n"
"background-color: rgb(0, 170, 0);")
        self.Sim_but.setObjectName("Sim_but")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 970, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        ##кнопка для темплейт метода
        self.Template_but.clicked.connect(self.showPicts)

        ##кнопка для виола  метода
        self.Viola_but.clicked.connect(self.showViola)

        ##кнопка для вывода симметрии
        self.Sim_but.clicked.connect(self.showSim)
        ##add items
        self.comboBox_2.addItems(res)

        ##add items
        self.comboBox.addItems(res2)
        ##Вывод изображение из выпадающего окна
        pixmap = QtGui.QPixmap(self.comboBox.currentText())
        pixmap2 = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio)
        self.tmplate.setPixmap(pixmap2)
        ##Вывод изображение из выпадающего окна
        self.pictures.setPixmap(QtGui.QPixmap(self.comboBox_2.currentText()))


        ##Вывод изображение из выпадающего окна
        self.comboBox.currentIndexChanged.connect(self.on_combobox_changed)
    def on_combobox_changed(self):
        pixmap = QtGui.QPixmap(self.comboBox.currentText())
        pixmap2 = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio)
        self.pictures.setPixmap(pixmap2)

        ##Вывод изображение из выпадающего окна
        self.comboBox_2.currentIndexChanged.connect(self.on_combobox_changed_2)
    def on_combobox_changed_2(self):
        pixmap = QtGui.QPixmap(self.comboBox_2.currentText())
        pixmap2 = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio)
        self.tmplate.setPixmap(pixmap2)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Template_but.setText(_translate("MainWindow", "Template метод"))
        self.result.setText(_translate("MainWindow", "Result"))
        self.tmplate.setText(_translate("MainWindow", "template"))
        self.pictures.setText(_translate("MainWindow", "pictures"))
        self.label_4.setText(_translate("MainWindow", "результат"))
        self.label_5.setText(_translate("MainWindow", "Шаблон"))
        self.label_6.setText(_translate("MainWindow", "Картинка"))
        self.label_7.setText(_translate("MainWindow", "Выбрать картинку 1"))
        self.label_8.setText(_translate("MainWindow", "Выбрать картинку 2"))
        self.Viola_but.setText(_translate("MainWindow", "Метод Виолы-Джонса"))
        self.Sim_but.setText(_translate("MainWindow", "Линии симметрии"))
    ##Показать темлейт метод
    def showPicts(self):
        self.template(self.comboBox_2.currentText(),self.comboBox.currentText())

    ##Показать  метод виолы джонсон
    def showViola(self):
        self.Viola_J(self.comboBox_2.currentText())

    def showSim(self):
        self.sim(self.comboBox_2.currentText())
    def template(self,temp,pic):
        img_rgb = cv.imread(pic)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        template = cv.imread(temp,0)
        # crop_img = template[40:95,10:80]
        w, h = template.shape[::-1]
        res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        # cv.imwrite('res.png',img_rgb)
        frame = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(image)
        pixmap2 = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio)
        self.result.setPixmap(pixmap2)

    def Viola_J(self, temp):

        face_cascade = cv.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
        eyes_cascade = cv.CascadeClassifier('cascade/haarcascade_eye.xml')
        smile_cascade = cv.CascadeClassifier('cascade/haarcascade_smile.xml')
        glass_cascade = cv.CascadeClassifier('cascade/haarcascade_eye_tree_eyeglasses.xml')
        img_rgb = cv.imread(temp)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
        eyes = eyes_cascade.detectMultiScale(img_gray, 1.3, 5)
        smile = smile_cascade.detectMultiScale(img_gray, 1.3, 5)
        glass = glass_cascade.detectMultiScale(img_gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # for (ex, ey, ew, eh) in eyes:
        #     cv.rectangle(img_rgb, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
        for (ex, ey, ew, eh) in smile:
            cv.rectangle(img_rgb,(ex,ey),(ex+ew,ey+eh), (255,0,0),2)
        for (ex, ey, ew, eh) in glass:
            cv.rectangle(img_rgb,(ex,ey),(ex+ew,ey+eh), (255,0,0),2)
        frame = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(image)
        pixmap2 = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio)
        self.result.setPixmap(pixmap2)

        ##Линии симметрии


    def sim(self,filename):
        test = []
        img_rgb = cv.imread(filename)
        detector = MTCNN()
        faces = detector.detect_faces(img_rgb)
        data = plt.imread(filename)
        plt.imshow(data)
        for result in faces:
            for key, value in result['keypoints'].items():
                if key == 'left_eye':
                    test.append(value)
                if key == 'right_eye':
                    test.append(value)
        x, y, width, height = result['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        new_array = [n for tup in test for n in tup]
        mid_x = (new_array[0] + new_array[2]) / 2
        mid_y = (new_array[1] + new_array[3]) / 2
        dist_x = math.sqrt((new_array[0] - mid_x)**2+(new_array[1] - mid_y)**2)
        ax = plt.gca()
        # горизонтальная
        l = ax.axline([new_array[0], new_array[1]], [new_array[2], new_array[3]])
        ax.add_patch(rect)
        # ##основная симметричная
        l2 = ax.axline([x + width/2,y + height], [mid_x, mid_y])
        # ##дополнительная симметричная
        # l3 = ax.axvline(new_nose[0] + dist_x)
        # l4 = ax.axvline(new_nose[0] - dist_x)
        l3 = ax.axline([new_array[0], new_array[1]],[x + width/2- dist_x,y + height])
        l4 = ax.axline([new_array[2], new_array[3]], [x + width / 2 + dist_x, y + height])
        ###prochee
        # l2 = ax.axline([new_nose[0], new_nose[1]], [mid_x, mid_y])
        # l2 = mlines.Line2D([new_nose[0], mid_x ], [new_nose[1], mid_y])
        ax.add_line(l)
        ax.add_line(l2)
        ax.add_line(l3)
        ax.add_line(l4)
        plt.show()
