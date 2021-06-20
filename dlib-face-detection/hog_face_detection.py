# USAGE
# python hog_face_detection.py --image images/HID_1.png

import helpers
import argparse
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
ap.add_argument("-u", "--upsample", type=int, default=1,
	help="# of times to upsample")
args = vars(ap.parse_args())


print("[INFO] loading HOG + Linear SVM face detector...")
detector = dlib.get_frontal_face_detector()

image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

start = time.time()

rects = detector(rgb, args["upsample"])
end = time.time()
print("[INFO] face detection took {:.4f} seconds".format(end - start))

boxes = [helpers.convert_and_trim_bb(image, r) for r in rects]

if len(boxes) == 0:
  print('Fasle')
else:
  print('True')

for (x, y, w, h) in boxes:
	
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.imshow(image)
