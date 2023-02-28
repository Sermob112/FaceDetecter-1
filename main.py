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
# # Read the main image
# img_rgb = cv2.imread('pic/test.jpg')
#
# # Convert it to grayscale
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#
# # Read the template
# template = cv2.imread('pic/matching.jpg',0)
# w, h = template.shape[::-1]
#
# # Perform match operations.
# res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
#
# # Specify a threshold
# threshold = 0.5
#
# # Store the coordinates of matched area in a numpy array
# loc = np.where(res >= threshold)
#
# # Draw a rectangle around the matched region.
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
#
# # Show the final image with the matched area.
# cv2.imshow('Detected', img_rgb)
# cv2.waitKey(0)

# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv.imread('pic/test.jpg',0)
# img2 = img.copy()
# template = cv.imread('pic/matching.jpg',0)
# w, h = template.shape[::-1]
# # All the 6 methods for comparison in a list
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
#             'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
# for meth in methods:
#     img = img2.copy()
#     method = eval(meth)
#     # Apply template Matching
#     res = cv.matchTemplate(img,template,method)
#     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
#     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#     if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv.rectangle(img,top_left, bottom_right, 255, 2)
#     plt.subplot(121),plt.imshow(res,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()
#
##Линии симметрии
# img_rgb = cv.imread('pic/test.jpg')
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
test = []
nose = []
def draw_image_with_boxes(filename, result_list):
    # load the image
    data = plt.imread(filename)
    # plot the image
    plt.imshow(data)
    # get the context for drawing boxes
    ax = plt.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        # rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        #draw line
        # line =  plt.plot(x, y)
        # ax.add_patch(rect)
        # draw the dots
        for key, value in result['keypoints'].items():

            # create and draw dot
            if key == 'left_eye':
                test.append(value)
            if key == 'right_eye':
                test.append(value)
            if key  == 'nose':
                nose.append(value)

            dot = Circle(value, radius=1, color='red')
            ax.add_patch(dot)
    new_array = [n for tup in test for n in tup]
    new_nose = [n for tup in nose for n in tup]
    #горизонтальная
    l = ax.axline([new_array[0], new_array[1]], [new_array[2], new_array[3]])
    # l = mlines.Line2D([new_array[0], new_array[2]], [new_array[1], new_array[3]])
    mid_x = (new_array[0] + new_array[2]) / 2
    mid_y = (new_array[1] + new_array[3]) / 2
    dist_x = math.sqrt((new_array[0] - mid_x)**2+(new_array[1] - mid_y)**2)
    ##основная симметричная
    l2 = ax.axline([new_nose[0], new_nose[1]], [mid_x, mid_y])
    ##дополнительная симметричная
    l3 = ax.axvline(new_nose[0] + dist_x)
    l4 = ax.axvline(new_nose[0] - dist_x)
    # l2 = ax.axline([new_nose[0], new_nose[1]], [mid_x, mid_y])
    # l2 = mlines.Line2D([new_nose[0], mid_x ], [new_nose[1], mid_y])
    # ax.add_line(l)
    ax.add_line(l2)
    ax.add_line(l3)
    ax.add_line(l4)

        # l = mlines.Line2D(test[0],test[1])
        # ax.add_line(l)
    # show the plot

    plt.show()
filename = 'pic/test.jpg'
# load image from file
pixels = plt.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
# display faces on the original image
draw_image_with_boxes(filename, faces)


###метод Виолы - Джонес

# face_cascade = cv.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
# eyes_cascade = cv.CascadeClassifier('cascade/haarcascade_eye.xml')
# smile_cascade = cv.CascadeClassifier('cascade/haarcascade_smile.xml')
# img_rgb = cv.imread('pic/test4.jpg')
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(img_gray,1.3,5)
# eyes = eyes_cascade.detectMultiScale(img_gray,1.3,5)
# smile = smile_cascade.detectMultiScale(img_gray,1.3,5)
# for (x, y, w, h) in faces:
#     cv.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
# for (ex, ey, ew, eh) in eyes:
#     cv.rectangle(img_rgb,(ex,ey),(ex+ew,ey+eh), (255,0,0),2)
# for (ex, ey, ew, eh) in smile:
#     cv.rectangle(img_rgb,(ex,ey),(ex+ew,ey+eh), (255,0,0),2)
# cv.imshow('img', img_rgb)
# cv.waitKey(0)
# cv.destroyAllWindows()

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

from matplotlib import pyplot as plt
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

## template метод
# img_rgb = cv.imread('pic/test4.jpg')
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
# template = cv.imread('pic/matching4.jpg',0)
# w, h = template.shape[::-1]
# res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
# threshold = 0.4
# loc = np.where( res >= threshold)
# for pt in zip(*loc[::-1]):
#     cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
# cv.imwrite('res.png',img_rgb)
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