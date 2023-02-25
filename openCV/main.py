# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
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


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

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

img_rgb = cv.imread('pic/test4.jpg')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('pic/matching4.jpg',0)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.4
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv.imwrite('res.png',img_rgb)
cv.imshow("img_rgb", img_rgb)
cv.waitKey(0)


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