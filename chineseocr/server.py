# coding=utf-8
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
from flask import Flask, request, jsonify
import io
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import copy
import json
import requests

app = Flask(__name__)
model = None
output = None


# AWS ubuntu
TestingDataPath = os.path.join(os.getcwd(),'images/testing-data')
ModelPath = os.path.join(os.getcwd(), 'models')
LabelNames = '拈拉拋拌拍拎拐拒拓拔拖拗拘拙拚招放斧於旺昀昂昆昌明昏易昔朋服'
ModelName = 'CNN_Model.h5'

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img

def img_preprocessing(image, width=50, height=50, local = True):
    if local:
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

@app.route('/', methods=['get'])
def hello():
    return 'hello !\n'

@app.route('/localPredict', methods=['post'])
def localPredict():
  os.chdir(TestingDataPath)
  if request.method == 'POST':
     if request.files.get('image'):
        image = request.files.get('image')
        #print('shape', image.shape)
        image = img_preprocessing(image, 50, 50)
        images = img_reshape(image)
        prediction = model.predict(images)
        result = showResult(prediction)
        print('result', result)
  return result + '\n'

@app.route('/predict', methods=['post'])
def predict():
  global output
  os.chdir(TestingDataPath)
  if request.method == 'POST': 
     data = request.json
     print('data', data)
     imageUrl = data['image']
     # perform request
     response =  requests.get(imageUrl).content
     # convert to array of ints
     nparr = np.frombuffer(response, np.uint8)
     # convert to image array
     image = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)
     top = data['top']
     local = False
     image = img_preprocessing(image, 50, 50, local)
     images = img_reshape(image)
     prediction = model.predict(images)
     result = showResult(prediction, top)
     output = json.loads(result)
     print('output', output)
  return jsonify(output)
  
if __name__ == '__main__':
  #os.chdir(ModelPath)
  print('ModelPath', ModelPath)
  print('ModelName', ModelName)
  model = load_model(os.path.join(ModelPath, ModelName))
  app.run(host="0.0.0.0", debug=True, port=5000)


