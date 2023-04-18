
import cv2 as cv2
import numpy as np

import os





def hist_correl(e,t):

    img1 = cv2.imread(e)
    img2 = cv2.imread(t)
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist([gray_img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray_img2], [0], None, [256], [0, 256])
    # Нормализация гистограмм
    cv2.normalize(hist1, hist1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    delt_h = abs(hist1 - hist2)
    g = np.array(delt_h)
    percent_diff = (np.sum(g) / np.prod(g.shape) * 100)
    # Сравнение гистограмм
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    match = 100 -  cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL) * 100
    percentage = round((match + 1) * 50,1)
    return round(match,3)
#Сцена меняется 6 раз

standards = []
def select_files(n = 200):
    files = []
    directory = "Photos"

    for files in os.listdir(directory)[:n]:
        standards.append(f'{directory}/{files}')
    return standards
# print(len(select_files()))

result = []

# print(hist_correl(standards[0], standards[1]))
def find_scene():
    j = 1
    data = {}
    for i in standards:
        try:
            result.append(hist_correl(i,standards[j]))
            data[hist_correl(i,standards[j])] = {i:standards[j]}
            j= j+ 1
        except Exception:
            data[hist_correl(i,standards[len(standards)-1])] = {i:standards[len(standards)-1]}
    return data
#Кадры изменения сцены 103 172 283

def scene_dect():
    scene_result = {}
    for proc, pics in find_scene().items():
        if proc > 10:
            scene_result[proc] = pics
    return scene_result

print(len(scene_dect()))
print(scene_dect())


