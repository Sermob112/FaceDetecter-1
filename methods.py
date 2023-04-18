import cv2 as cv2
import numpy as np

import os
###########################################
#чтение всех файлов
file_names = []
path = 'orl_faces'
folders = os.listdir(path)
etalon = []
test = []
result  = []
files = []
i = 1
######################################################################################################
#Сравнение двух гистограмм

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

    # Сравнение гистограмм
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    match = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL) * 100
    percentage = round((match + 1) * 50,1)
    return match
######################################################################################################
#Сравнение двух изображений методом DCT

def dct_correl(e,t):
    img1 = cv2.imread(e, 0)
    img2 = cv2.imread(t, 0)

    # применяем дискретное косинусное преобразование (DCT)
    dct1 = cv2.dct(np.float32(img1))
    dct2 = cv2.dct(np.float32(img2))
    # Вычисление модуля спектра
    # mag1 = cv2.magnitude(dct1, dct1)
    # mag2 = cv2.magnitude(dct2, dct2)
    from scipy.spatial.distance import cosine

    similarity_score = 1 - cosine(dct1.flatten(), dct2.flatten())

    # # вычисляем разницу между двумя DCT-преобразованиями
    # diff = cv2.absdiff(mag1, mag2)
    # score = np.sum(diff) / np.sum(mag1) * 100
    #
    # # вычисление меры расстояния
    # dist = np.linalg.norm(dct1 - dct2)
    # max_dist = np.sqrt(img1.shape[0] * img1.shape[1]) * 255
    # similarity = (1 - (dist / max_dist)) * 100
    return similarity_score * 100
######################################################################################################
#Сравнение двух изображений методом DFT

def dft_correl(e,t):

    # Загрузите два изображения
    img1 = cv2.imread(e,0)
    img2 = cv2.imread(t,0)
    dft1 = cv2.dft(np.float32(img1), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(np.float32(img2), flags=cv2.DFT_COMPLEX_OUTPUT)

    # вычисление меры расстояния
    dist = np.linalg.norm(dft1 - dft2)

    #нормализация расстояния
    max_dist = np.sqrt(img1.shape[0] * img1.shape[1] * 2) * 255
    similarity = ((max_dist / dist)) * 100
    return  100 - similarity
######################################################################################################
#Сравнение двух изображений по градиенту
def grad_correl(e,t):
    img1 = cv2.imread(e)
    img2 = cv2.imread(t)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    grad_x1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
    grad_y1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag1 = cv2.magnitude(grad_x1, grad_y1)

    grad_x2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
    grad_y2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag2 = cv2.magnitude(grad_x2, grad_y2)

    # Calculate the mean of the gradient magnitude of each image
    mean_grad_mag1 = cv2.mean(grad_mag1)[0]
    mean_grad_mag2 = cv2.mean(grad_mag2)[0]

    # Compare the mean of the gradient magnitude of each image and output the result as a percentage
    if mean_grad_mag1 > mean_grad_mag2:
        similarity = mean_grad_mag2 / mean_grad_mag1 * 100
    else:
        similarity = mean_grad_mag1 / mean_grad_mag2 * 100

    return similarity
######################################################################################################
#Сравнение двух изображений по масштабу
def scale_correl(e,t):
    from skimage import io, measure, transform,metrics
    img1 = io.imread(e)
    img2 = io.imread(t)

    image1_resized = transform.resize(img1, (12, 14))
    image2_resized = transform.resize(img2, (12, 14))

    # Сравнение двух изображений с использованием функции structural_similarity
    ssim = metrics.structural_similarity(image1_resized, image2_resized, data_range= 255) * 100

    return  ssim

#############################################################################################
#чтение из файла и добаление в массивы эталонов и тестов
def read_Etalon_and_test(b):
    for folder in folders:
        for i in range(b+1,11):
            test.append(f'orl_faces/{folder}/{i}.pgm')
        for j in range(1,b+1):
            etalon.append(f'orl_faces/{folder}/{j}.pgm')

#############################################################################################
#Срвнение всех методов
def Finder(b):
    k = 0
    read_Etalon_and_test(b)
    for i in range(len(etalon)):
        for l in range(b):
            for j in range(10 - b):
                result.append(round(hist_correl(etalon[i],test[j +k]),1))
                result.append(round(dft_correl(etalon[i], test[j +k]),1))
                result.append(round(dct_correl(etalon[i], test[j +k]),1))
                result.append(round(grad_correl(etalon[i], test[j +k]),1))
                result.append(round(scale_correl(etalon[i], test[j +k]),1))

        k = 10 - b

    return result
#############################################################################################################
#Кросс валидация###


all_imges= []
def test_for_cross():
    for folder in folders:
        for i in range(1,11):
                all_imges.append(f'orl_faces/{folder}/{i}.pgm')


# read_Etalon_and_test_for_cross()

standards = []
tests = []
#Выбрать кастомное количество тестов и эталонов
def select_files(n = 10, k = 1):
    files = []
    directory = "orl_faces"

    for class_folder in os.listdir('orl_faces')[:n]:
        class_path = os.path.join('orl_faces', class_folder)
        if os.path.isdir(class_path):
            even_files = []
            odd_files = []
            for i, file_name in enumerate(os.listdir(class_path)):
                file_path = os.path.join(class_path, file_name)
                if i % 2 == 0 and len(even_files) < k:
                    even_files.append(file_path)
                else:
                    odd_files.append(file_path)
            standards.extend(even_files)
            tests.extend(odd_files)


hist_result = []
dct_result = []
dft_result = []
grad_result = []
scale_result = []
def final_validation(k = 10, n = 2):
    select_files(k, n)
    v = 0
    h = 0
    for i in range(len(standards)):
        for k in range(10 - n):
            hist_result.append(round(hist_correl(standards[i],tests[k + v]),0))
            dct_result.append(round(dct_correl(standards[i], tests[k + v]), 0))
            dft_result.append(round(dft_correl(standards[i], tests[k + v]), 0))
            grad_result.append(round(grad_correl(standards[i], tests[k + v]), 0))
            scale_result.append(round(scale_correl(standards[i], tests[k + v]), 0))


def result_Hist():
    return hist_result
def result_DCT():
    return hist_result
def result_():
    return hist_result
def result_Hist():
    return hist_result
def result_Hist():
    return hist_result


# final_validation(10,2)
# print(len(hist_result))



# fawkes.Fawkes('matching.jpg', gpu= 1, batch_size=1,mode="low", format= 'png')
# protected_image = clip_img(f)
# # Save the image using OpenCV
# cv2.imwrite('protected_image.jpg', protected_image)

# img = Image.open('Orl_faces/s1/1.pgm')
# pprotected_img = fawkes.shield(img)
# pprotected_img.save('test.jpg')
###################################################################################################

# test_for_cross()
#
# etalon_train= []
# test_train = []
#
# n_samples = 4
# scores = []
# def croos_valid():
#     pairs = list(product(all_imges, repeat=1))
#     for train_index, test_index in KFold(n_splits=4, shuffle=True).split(pairs):
#         # выбираем только те пары, которые попали в тестовый фолд
#         test_pairs = [pairs[i] for i in test_index]
#         # считаем метрику качества для каждой пары
#         fold_scores = []
#         for pair in test_pairs:
#             # загрузка изображений и применение dct
#             image1 = cv2.imread(pair[0],0)
#             image2 = cv2.imread(pair[1],0)
#             dct1 = cv2.dct(np.float32(image1))
#             dct2 = cv2.dct(np.float32(image2))
#             # сравнение изображений и вычисление метрики качества
#             dist = np.linalg.norm(dct1 - dct2)
#             score = (1 - dist / (image1.shape[0] * image1.shape[1])) * 100
#             fold_scores.append(score)
#         # вычисление среднего значения метрики качества для текущего фолда
#         mean_score = sum(fold_scores) / len(fold_scores)
#         scores.append(mean_score)
#
#     # вычисляем среднее значение метрики качества по всем фолдам
#     mean_accuracy = sum(scores) / len(scores)
#     return mean_accuracy
# print(croos_valid())
####################################################
#TEST
#
# read_Etalon_and_test(1)
# result = Finder(2)
# print(len(result)//5)
# print(len(test))
# print(len(etalon))
#
# print(test)
# print(etalon)
# print(Hist_correl(etalon[1],test[1]))
# print(DFT_correl(etalon[0],test[1]))
# print(DCT_correl(etalon[1],test[3]))
# print(Grad_correl(etalon[1],test[155]))
# print(Scale_correl(etalon[1],test[20]))


# app = QtWidgets.QApplication([])
# window = MainWindow()
# window.show()
# app.exec_()




# for folder in folders:
#     folder_path = os.path.join(path, folder)
#     if os.path.isdir(folder_path):
#         files = os.listdir(folder_path)
#         for file in files:
#             file_names.append(os.path.join(folder_path, file))
# img = cv.imread(file_names[])
# cv.imshow('Fourier Transform', img)
# cv.waitKey(0)


# class mywindow(QtWidgets.QMainWindow):
#     def __init__(self):
#         super(mywindow, self).__init__()
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)
#
# def application():
#     app = QApplication(sys.argv)
#     window = mywindow()
#
#     window.show()
#     sys.exit(app.exec_())
# if __name__ == '__main__':
#     application()


# ####Гистограмма
# img = cv.imread('s12/1.jpg')
# # calculate mean value from RGB channels and flatten to 1D array
# vals = img.mean(axis=2).flatten()
# # plot histogram with 255 bins
# b, bins, patches = plt.hist(vals, 255)
# plt.xlim([0,255])
# plt.show()
# ####DFT
#
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# # Compute the discrete Fourier Transform of the image
# fourier = cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)
#
# # Shift the zero-frequency component to the center of the spectrum
# fourier_shift = np.fft.fftshift(fourier)
#
# # calculate the magnitude of the Fourier Transform
# magnitude = 20 * np.log(cv.magnitude(fourier_shift[:, :, 0], fourier_shift[:, :, 1]))
#
# # Scale the magnitude for display
# magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
#
# # Display the magnitude of the Fourier Transform
# cv.imshow('Fourier Transform', magnitude)
# cv.waitKey(0)

###Градиент

# sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
# sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
#
# # Вычисление абсолютного значения градиента
# abs_sobelx = cv.convertScaleAbs(sobelx)
# abs_sobely = cv.convertScaleAbs(sobely)
#
# # Объединение градиентов по горизонтали и вертикали
# grad = cv.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
#
# # Отображение изображения с градиентом
# cv.imshow('Gradient', grad)
# cv.waitKey(0)
# cv.destroyAllWindows()


# ###DCT
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# # Применяем DCT
# dct = cv.dct(np.float32(gray) / 255.0)
#
# # Обратное DCT
# idct = cv.idct(dct) * 255.0
#
# # Отображение результатов
# cv.imshow('Original Image', img)
# cv.imshow('DCT', dct)
# cv.imshow('IDCT', idct)
# cv.waitKey(0)
# cv.destroyAllWindows()

# imgcv1 = cv.split(img)[0]
# imf = np.float32(imgcv1)/255.0  # float conversion/scale
# dct = cv.dct(imf)              # the dct
# imgcv1 = np.uint8(dct*255.0)
#
#
# cv.imshow('DCT', imgcv1)
# cv.waitKey(0)




























# def sim(filename):
#     test = []
#     img_rgb = cv.imread(filename)
#     detector = MTCNN()
#     faces = detector.detect_faces(img_rgb)
#     data = plt.imread(filename)
#     plt.imshow(data)
#     for result in faces:
#         for key, value in result['keypoints'].items():
#             if key == 'left_eye':
#                 test.append(value)
#             if key == 'right_eye':
#                 test.append(value)
#     x, y, width, height = result['box']
#     rect = Rectangle((x, y), width, height, fill=False, color='red')
#     new_array = [n for tup in test for n in tup]
#     mid_x = (new_array[0] + new_array[2]) / 2
#     mid_y = (new_array[1] + new_array[3]) / 2
#     dist_x = math.sqrt((new_array[0] - mid_x)**2+(new_array[1] - mid_y)**2)
#     ax = plt.gca()
#     # горизонтальная
#     l = ax.axline([new_array[0], new_array[1]], [new_array[2], new_array[3]])
#     ax.add_patch(rect)
#     # ##основная симметричная
#     l2 = ax.axline([x + width/2,y + height], [mid_x, mid_y])
#     # ##дополнительная симметричная
#     # l3 = ax.axvline(new_nose[0] + dist_x)
#     # l4 = ax.axvline(new_nose[0] - dist_x)
#     l3 = ax.axline([new_array[0], new_array[1]],[x + width/2- dist_x,y + height])
#     l4 = ax.axline([new_array[2], new_array[3]], [x + width / 2 + dist_x, y + height])
#     ###prochee
#     # l2 = ax.axline([new_nose[0], new_nose[1]], [mid_x, mid_y])
#     # l2 = mlines.Line2D([new_nose[0], mid_x ], [new_nose[1], mid_y])
#     ax.add_line(l)
#     ax.add_line(l2)
#     ax.add_line(l3)
#     ax.add_line(l4)
#     plt.show()
# sim('s12/1.jpg')
# #############################################################################
#template метод
# def template(temp, pic):
#     img_rgb = cv.imread(pic)
#     img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
#     template = cv.imread(temp,0)
#     crop_img = template[40:95,10:80]
#     w, h = template.shape[::-1]
#     res = cv.matchTemplate(img_gray,crop_img,cv.TM_CCOEFF_NORMED)
#     threshold = 0.99
#     loc = np.where( res >= threshold)
#     for pt in zip(*loc[::-1]):
#         cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
#     # cv.imwrite('res.png',img_rgb)
#
#     cv.imshow("Template", crop_img)
#     cv.imshow("img_rgb", img_rgb)
#     cv.waitKey(0)
# # template('s12/1.pgm','s12/s12_all.pbm')
# #######################################################################
#
# ###метод Виолы - Джонес
# def Viola_J(filename):
#
#     face_cascade = cv.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
#     eyes_cascade = cv.CascadeClassifier('cascade/haarcascade_eye.xml')
#     smile_cascade = cv.CascadeClassifier('cascade/haarcascade_smile.xml')
#     glass_cascade = cv.CascadeClassifier('cascade/haarcascade_eye_tree_eyeglasses.xml')
#     img_rgb = cv.imread(filename)
#     img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(img_gray,1.3,5)
#     eyes = eyes_cascade.detectMultiScale(img_gray,1.3,5)
#     smile = smile_cascade.detectMultiScale(img_gray,1.3,5)
#     glass = glass_cascade.detectMultiScale(img_gray,1.3,5)
#     # for (x, y, w, h) in faces:
#     #     cv.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
#     for (ex, ey, ew, eh) in eyes:
#         cv.rectangle(img_rgb,(ex,ey),(ex+ew,ey+eh), (255,0,0),2)
#     # for (ex, ey, ew, eh) in smile:
#     #     cv.rectangle(img_rgb,(ex,ey),(ex+ew,ey+eh), (255,0,0),2)
#     # for (ex, ey, ew, eh) in glass:
#     #     cv.rectangle(img_rgb,(ex,ey),(ex+ew,ey+eh), (255,0,0),2)
#     cv.imshow('img', img_rgb)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
# # Viola_J('s12/s27_all.jpg')
# ##Линии симметрии
# # img_rgb = cv.imread('pic/test.jpg')
# # img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
# test = []
# nose = []
# def draw_image_with_boxes(filename, result_list):
#     data = plt.imread(filename)
#     plt.imshow(data)
#     for result in result_list:
#         for key, value in result['keypoints'].items():
#             if key == 'left_eye':
#                 test.append(value)
#             if key == 'right_eye':
#                 test.append(value)
#     x, y, width, height = result['box']
#     rect = Rectangle((x, y), width, height, fill=False, color='red')
#     new_array = [n for tup in test for n in tup]
#     mid_x = (new_array[0] + new_array[2]) / 2
#     mid_y = (new_array[1] + new_array[3]) / 2
#     dist_x = math.sqrt((new_array[0] - mid_x)**2+(new_array[1] - mid_y)**2)
#     ax = plt.gca()
#     # горизонтальная
#     l = ax.axline([new_array[0], new_array[1]], [new_array[2], new_array[3]])
#     ax.add_patch(rect)
#     # ##основная симметричная
#     l2 = ax.axline([x + width/2,y + height], [mid_x, mid_y])
#     # ##дополнительная симметричная
#     # l3 = ax.axvline(new_nose[0] + dist_x)
#     # l4 = ax.axvline(new_nose[0] - dist_x)
#     l3 = ax.axline([new_array[0], new_array[1]],[x + width/2- dist_x,y + height])
#     l4 = ax.axline([new_array[2], new_array[3]], [x + width / 2 + dist_x, y + height])
#     ###prochee
#     # l2 = ax.axline([new_nose[0], new_nose[1]], [mid_x, mid_y])
#     # l2 = mlines.Line2D([new_nose[0], mid_x ], [new_nose[1], mid_y])
#     ax.add_line(l)
#     ax.add_line(l2)
#     ax.add_line(l3)
#     ax.add_line(l4)
#     plt.show()
# for i in range(1,8):
#     filename = f's12/{8}.pgm'
#     img_rgb = cv.imread(filename)
#     img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
#     # создать детектор, используя веса по умолчанию
#     detector = MTCNN()
#     # распознать лица на изображении
#     faces = detector.detect_faces(img_rgb)
#     # отобразить лица на исходном изображении
#     draw_image_with_boxes(filename, faces)


##############################################################################################################
##дополнительная линия связывающая глаза

# eyes_cascade = cv.CascadeClassifier('cascade/haarcascade_eye.xml')
# img_rgb = cv.imread('pic/matching3.jpg')
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
# plt.imshow(img_rgb[:, :, ::-1])
# ax = plt.gca()
# eyes = eyes_cascade.detectMultiScale(img_gray)
#
# eye = eyes[:, 2]
# container1 = []
# for i in range(0, len(eye)):
#     container = (eye[i], i)
#     container1.append(container)
# df = pd.DataFrame(container1, columns=[
#     "length", "idx"]).sort_values(by=['length'])
# eyes = eyes[df.idx.values[0:2]]
#
# # deciding to choose left and right eye
# eye_1 = eyes[0]
# eye_2 = eyes[1]
# if eye_1[0] > eye_2[0]:
#     left_eye = eye_2
#     right_eye = eye_1
# else:
#     left_eye = eye_1
#     right_eye = eye_2
#
# # center of eyes
# # center of right eye
# right_eye_center = (
#     int(right_eye[0] + (right_eye[2] / 2)),
#     int(right_eye[1] + (right_eye[3] / 2)))
# right_eye_x = right_eye_center[0]
# right_eye_y = right_eye_center[1]
#
#
# # center of left eye
# left_eye_center = (
#     int(left_eye[0] + (left_eye[2] / 2)),
#     int(left_eye[1] + (left_eye[3] / 2)))
# left_eye_x = left_eye_center[0]
# left_eye_y = left_eye_center[1]
# l = mlines.Line2D([right_eye_x,left_eye_x], [right_eye_y,left_eye_y])
# ax.add_line(l)
#
# symmetry_line = cv.HoughLinesP(img_gray, 1, np.pi/180, 100)
# for x1,y1,x2,y2 in symmetry_line[0]:
#     cv.line(img_rgb, (x1,y1), (x2,y2), (0,255,0), 2)
# local_lines = cv.HoughLinesP(img_gray, 1, np.pi/180, 50)
# for x1,y1,x2,y2 in local_lines[0]:
#     cv.line(img_rgb, (x1,y1), (x2,y2), (255,0,0), 2)
# plt.imshow(img_rgb)
# plt.show()



#Для видеокамеры
# def detect(gray, frame):
#     faces = face_cascade.detectMultiScale(gray,1.3,5)
#     for (x,y,w,h) in faces:
#         cv.rectangle(frame,(x,y),(x+h), (255,0,0),2)
#         roi_gray = gray[y:y+h,x:x+w]
#         roi_frame = frame[y:y+h,x:x+w]
#         eyes = eyes_cascade.detectMultiScale(roi_gray,1.1,5)
#         for(ex,ey,ew,eh) in eyes:
#             cv.rectangle(roi_frame,(ex,ey),(ex+ew,ey+eh), (255,0,0),2)
#             cv.imshow('Findframe',frame)
#             if cv.waitKey(0) &0xFF == ord('q'):
#                 break
# img_rgb = cv.imread('pic/test4.jpg')
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
# detecting_face = (img_gray,img_gray)
# cv.imshow("img_rgb", img_rgb)
# cv.waitKey(0)

## template метод
# import cv2 as cv
# import numpy as np
# img_rgb = cv.imread('pic/test4.jpg')
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
#
# template = cv.imread('pic/matching.jpg',0)
# w, h = template.shape[::-1]
# res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
# threshold = 0.4
# loc = np.where( res >= threshold)
# for pt in zip(*loc[::-1]):
#     cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
# # cv.imwrite('res.png',img_rgb)
# cv.imshow("img_rgb", img_rgb)
# cv.waitKey(0)

##нейросеть по поиску лица
# import cv2
# import numpy as np
#
# # https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
# prototxt_path = "weights/deploy.prototxt.txt"
# # https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
# model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
#
# # load Caffe model
# model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
#
# # read the desired image
# image = cv2.imread("test.jpg")
# # get width and height of the image
# h, w = image.shape[:2]
#
# # preprocess the image: resize and performs mean subtraction
# blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
# # set the image into the input of the neural network
# model.setInput(blob)
# # perform inference and get the result
# output = np.squeeze(model.forward())
# font_scale = 1.0
# for i in range(0, output.shape[0]):
#     # получить уверенность
#     confidence = output[i, 2]
#     # если достоверность выше 50%, то нарисуйте окружающий прямоугольник
#     if confidence > 0.5:
#         # получить координаты окружающего блока и масштабировать их до исходного изображения
#         box = output[i, 3:7] * np.array([w, h, w, h])
#         # преобразовать в целые числа
#         start_x, start_y, end_x, end_y = box.astype(np.int32)
#         # рисуем прямоугольник вокруг лица
#         cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
#         # также нарисуем текст
#         cv2.putText(image, f"{confidence*100:.2f}%", (start_x, start_y-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)
# # show the image
# cv2.imshow("image", image)
# cv2.waitKey(0)
# # save the image with rectangles
# cv2.imwrite("kids_detected_dnn.jpg", image)