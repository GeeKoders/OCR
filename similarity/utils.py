import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import config
from sklearn.model_selection import train_test_split
from siamese_network import *

def get_image_data(imagePaths, label):
    data = []
    labels = []
    for image_name in os.listdir(imagePaths):
        image = os.path.join(imagePaths, image_name)
        image = cv2.imread(image)
        image = cv2.resize(image, config.IMG_SHAPE)
        data.append(image)
        labels.append(label)
    data = np.array(data, dtype = 'float32')
    data /= 255.0
    labels = np.array(labels)
    return data, labels

def load_train_data(file_path):
    fileList = []
    for i in os.listdir(file_path):
        fileList.append(i)
    data = []
    labels = []
    for i in range(len(fileList)):
        fileDir = fileList[i]
        allPath = os.path.join(file_path, fileList[i])
        data_i, labels_i = get_image_data(allPath, fileDir)
        data_i, labels_i = list(data_i), list(labels_i)
        data.extend(data_i)
        labels.extend(labels_i)
    data, labels = np.array(data), np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test, fileList

def load_data():
    
    x_train, x_test, y_train, y_test, filelist = load_train_data(config.TRAINING_PATH)
    print(x_train.shape, y_train.shape) 
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    input_shape = x_train.shape[1:]
    print(input_shape)
    print(filelist)
    digit_indices = [np.where(y_train == filelist[i])[0] for i in range(config.NUM_CLASSES)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)
    digit_indices = [np.where(y_test == filelist[i])[0] for i in range(config.NUM_CLASSES)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)
    print(te_pairs.shape, te_y.shape)
    return input_shape, tr_pairs, tr_y, te_pairs, te_y

def plot_training(H, plotPath):
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)