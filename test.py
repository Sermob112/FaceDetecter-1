import sys
import cv2 as cv
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import main
data = {'Имя': ['Иван', 'Наташа', 'Петр', 'Олег'],
        'Возраст': [23, 32, 45, 19],
        'Город': ['Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург']}

# Создаем датафрейм на основе массива
df = pd.DataFrame(data)

# Сохраняем датафрейм в виде таблицы Excel
df.to_excel('мой_файл.xlsx', index=False)