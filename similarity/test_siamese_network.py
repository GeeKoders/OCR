import cv2
import os
import numpy as np
import shutil
import imutils
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import PIL.ImageOps
import random

def get_image_data(imagePaths, label):
    data = []
    labels = []
    for image_name in os.listdir(imagePaths):
        image = os.path.join(imagePaths, image_name)
        image = cv2.imread(image)
        image = cv2.resize(image, (200, 200))
        data.append(image)
        labels.append(label)
    data = np.array(data, dtype = 'float')
    data /= 255.0
    labels = np.array(labels)
#     data, labels = shuffle(data, labels, random_state=42)
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

#from __future__ import absolute_import
#from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

num_classes = 2
epochs = 50


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

# num_classes = 2
print('cwd:', os.getcwd())

base = os.getcwd()

def load_data():
    x_train, x_test, y_train, y_test, filelist = load_train_data(os.path.join(base, 'training-data'))
    print(x_train.shape, y_train.shape)
#     num_classes = 2
    print('num_classes', num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    input_shape = x_train.shape[1:]  # (80, 80)
    print(input_shape)
    print(filelist)
    digit_indices = [np.where(y_train == filelist[i])[0] for i in range(num_classes)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)
    digit_indices = [np.where(y_test == filelist[i])[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)
    print(te_pairs.shape, te_y.shape)  # (980, 2, 80, 80) (980,)
    return input_shape, tr_pairs, tr_y, te_pairs, te_y

input_shape, tr_pairs, tr_y, te_pairs, te_y = load_data()

# network definition
base_network = create_base_network(input_shape)
# base_network = create_deep_network(input_shape, 2)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
