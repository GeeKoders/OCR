# coding=utf-8
from tensorflow.keras.models import load_model
from preprocessing_2375 import img_preprocessing, img_reshape
from decimal import Decimal
import os
import argparse
import copy
import json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input test image")
ap.add_argument("-t", "--top", type=int, required=False, default=3,
	help="path to select the top number chinese character")
args = vars(ap.parse_args())

print(args['top'])

def showResult(prediction, top = args['top']):
    nums = []
    for i in prediction.tolist():
        nums = i
    t = copy.deepcopy(nums)
    dict = {}
    for _ in range(args['top']):
        number = max(t)
        if number!= 0:
            index = t.index(number)
            t[index] = 0
            dict[index] = str(round(Decimal(str(number)),2)) + ',' + LabelNames[index]
    t = []
    jsonData = json.dumps(dict, ensure_ascii=False).encode('utf8')
    return jsonData.decode()

def labelOutput(File, LabelNames):
  with open(File, "r", encoding='big5') as f:
    for line in f.readlines():
      line = line.strip('\n')
      LabelNames.append(line)
  return LabelNames

print(os.getcwd())
File = os.path.join(os.getcwd(), 'images/training-data/training-data.txt')
TestingDataPath = os.path.join(os.getcwd(),'images/testing-data')
ModelPath = os.path.join(os.getcwd(), 'models')
LabelNames = []
ModeName = 'CNN_Model_2375.h5'

LabelNames = labelOutput(File, LabelNames)

os.chdir(ModelPath)
model = load_model(ModeName)

os.chdir(TestingDataPath)

image = img_preprocessing(args, args['image'])
images = img_reshape(image)

prediction = model.predict(images)
maxProbIdx = np.argmax(prediction)
print('predict value:', LabelNames[maxProbIdx])
result = showResult(prediction, args['top'])

print('Top:', args['top'])
print('result:', result)
