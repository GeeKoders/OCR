import cv2
import os
import numpy as np
import imutils
import random
import siamese_network
import config
import matplotlib.pyplot as plt
from imutils.paths import list_images
from keras.models import Model, load_model


testImagePaths = list(list_images(config.TESTING_PATH))
np.random.seed(42)
pairs = np.random.choice(testImagePaths, size=(4, 2))

model = load_model(os.path.join(config.MODEL_PATH,config.MODEL_NAME), custom_objects={'contrastive_loss': siamese_network.contrastive_loss})

for (i, (pathA, pathB)) in enumerate(pairs):

    imageA = cv2.imread(pathA)
    imageB = cv2.imread(pathB)

    origA = imageA.copy()
    origB = imageB.copy()

    imageA = np.expand_dims(imageA, axis=0)
    imageB = np.expand_dims(imageB, axis=0)


    # scale the pixel values to the range of [0, 1]
    imageA = imageA / 255.0
    imageB = imageB / 255.0

    # use our siamese model to make predictions on the image pair,
    # indicating whether or not the images belong to the same class
    preds = model.predict([imageA, imageB])
    print('preds:', preds)
    proba = preds[0][0]
    print('proba:', proba)
    # initialize the figure
    fig = plt.figure("Pair #{}".format(i + 1), figsize=(4, 2))
    plt.suptitle("Distance: {:.2f}".format(proba))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(origA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(origB, cmap=plt.cm.gray)
    plt.axis("off")
 
    print('imageA:', pathA)
    print('imageB:', pathB)
    if proba < 0.5:
       print("It is the same group")
    else:
       print("It is not the same group")
    
    # show the plot
    plt.show()