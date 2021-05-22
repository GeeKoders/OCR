# coding=utf-8
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
from flask import Flask, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import copy
import json
import requests
import io

app = Flask(__name__)
model = None

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img

def img_preprocessing(image, width=50, height=50):
    image = cv_imread(image)
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)[1]
    x, y, w, h = cv2.boundingRect(thresh)
    crop_image = image[y:y + h, x:x + w]
    image = cv2.resize(crop_image, (50, 50))
    return image
	
def img_reshape(image):
  images = []
  x=img_to_array(image)
  images.append(x)
  images=np.array(images)
  return images

def showResult(prediction, top = 3):
    nums = []
    for i in prediction.tolist():
        nums = i
    t = copy.deepcopy(nums)
    dict = {}
    for _ in range(top):
        number = max(t)
        if number!= 0:
            index = t.index(number)
            t[index] = 0
            dict[index] = str(number) + ',' + LabelNames[index]
    t = []
    jsonData = json.dumps(dict, ensure_ascii=False).encode('utf8')
    return jsonData.decode()

@app.route('/predict', methods=['post'])
def predict():
  os.chdir(TestingDataPath)
  if request.method == 'POST':
     if request.files.get('image'):
        image = request.files.get('image')
        image = img_preprocessing(image)
        images = img_reshape(image)

        prediction = model.predict(images)

        #top = request.files.get('top')
 
        result = showResult(prediction, 3)
        print('result', result)
  return result + '\n'
  
if __name__ == '__main__':
  #os.chdir(ModelPath)
  print('ModelPath', ModelPath)
  print('ModelName', ModelName)
  model = load_model(os.path.join(ModelPath, ModelName))
  app.run(host="0.0.0.0", debug=True, port=5000)


