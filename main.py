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
# #############################################################################
#template метод
def template(temp, pic):
    img_rgb = cv.imread(pic)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imread(temp,0)
    crop_img = template[40:95,10:80]
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img_gray,crop_img,cv.TM_CCOEFF_NORMED)
    threshold = 0.99
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    # cv.imwrite('res.png',img_rgb)

    cv.imshow("Template", crop_img)
    cv.imshow("img_rgb", img_rgb)
    cv.waitKey(0)
# template('s12/1.pgm','s12/s12_all.pbm')
#######################################################################

###метод Виолы - Джонес
def Viola_J(filename):

    face_cascade = cv.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
    eyes_cascade = cv.CascadeClassifier('cascade/haarcascade_eye.xml')
    smile_cascade = cv.CascadeClassifier('cascade/haarcascade_smile.xml')
    glass_cascade = cv.CascadeClassifier('cascade/haarcascade_eye_tree_eyeglasses.xml')
    img_rgb = cv.imread(filename)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray,1.3,5)
    eyes = eyes_cascade.detectMultiScale(img_gray,1.3,5)
    smile = smile_cascade.detectMultiScale(img_gray,1.3,5)
    glass = glass_cascade.detectMultiScale(img_gray,1.3,5)
    # for (x, y, w, h) in faces:
    #     cv.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(img_rgb,(ex,ey),(ex+ew,ey+eh), (255,0,0),2)
    # for (ex, ey, ew, eh) in smile:
    #     cv.rectangle(img_rgb,(ex,ey),(ex+ew,ey+eh), (255,0,0),2)
    # for (ex, ey, ew, eh) in glass:
    #     cv.rectangle(img_rgb,(ex,ey),(ex+ew,ey+eh), (255,0,0),2)
    cv.imshow('img', img_rgb)
    cv.waitKey(0)
    cv.destroyAllWindows()
# Viola_J('s12/s27_all.jpg')
##Линии симметрии
# img_rgb = cv.imread('pic/test.jpg')
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
test = []
nose = []
def draw_image_with_boxes(filename, result_list):
    data = plt.imread(filename)
    plt.imshow(data)
    for result in result_list:
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
for i in range(1,8):
    filename = f's12/{8}.pgm'
    img_rgb = cv.imread(filename)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    # создать детектор, используя веса по умолчанию
    detector = MTCNN()
    # распознать лица на изображении
    faces = detector.detect_faces(img_rgb)
    # отобразить лица на исходном изображении
    draw_image_with_boxes(filename, faces)


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