from tensorflow.keras.models import load_model
from preprocessing import img_preprocessing, img_reshape
import os
import argparse
import copy
import json

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
            dict[index] = str(number) + ',' + LabelNames[index]
    t = []
    jsonData = json.dumps(dict, ensure_ascii=False).encode('utf8')
    return jsonData.decode()


TestingDataPath = '/content/chineseocr/images/testing-data'
ModelPath = '/content/chineseocr/models'
LabelNames = '拈拉拋拌拍拎拐拒拓拔拖拗拘拙拚招放斧於旺昀昂昆昌明昏易昔朋服'
ModeName = 'CNN_Model.h5'

os.chdir(ModelPath)
model = load_model(ModeName)

os.chdir(TestingDataPath)

image = img_preprocessing(args, args['image'])
images = img_reshape(image)

print(images.shape)

prediction = model.predict(images)

result = showResult(prediction)

print(result)

