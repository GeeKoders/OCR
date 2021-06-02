# coding=utf-8
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import argparse


def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img

def img_preprocessing(args, image, width=50, height=50):
    image = cv_imread(image)
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)[1]
    x, y, w, h = cv2.boundingRect(thresh)
    crop_image = image[y:y + h, x:x + w]
    image = cv2.resize(crop_image, (50, 50))
    cv2.imwrite(args['image'], image)
    image = cv2.imread(args['image'])
    return image

def img_reshape(image):
  images = []
  x=img_to_array(image)
  images.append(x)
  images=np.array(images)
  images /= 255
  return images
