import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.ImageOps
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

def padding(img,size,color=(255,255,255)):
    w,h=img.size #col , row
    target_w , target_h = size
    new_img=PIL.ImageOps.expand(img,((target_w-w)//2,(target_h-h)//2),color).resize(size)
    return new_img
	
def resize_aspect_ratio(img,size): 
    w,h=img.size #col , row
    target_w , target_h = size
    ratio=min(target_w/w , target_h/h)
    new_img=img.resize( (int(w*ratio),int(h*ratio)) )
    return new_img
	
datagen = ImageDataGenerator(
    zca_whitening=False,
    rotation_range=5,
    width_shift_range=0.02,
    height_shift_range=0.02,
    shear_range=0.02,
    zoom_range=0.02,
    horizontal_flip=False,
    fill_mode='nearest')
	
print(os.getcwd())
preprocessing_path = os.path.join(os.getcwd(), 'preprocessing-data/HID')
os.chdir(preprocessing_path)
image = cv2.imread('HID_1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.reshape((1,) + image.shape)
print(image.shape)

i = 0
for batch in datagen.flow(image, batch_size=10,
          save_to_dir=os.getcwd(), save_prefix='ID', save_format='png'):
    plt.subplot(5,4,1 + i)
    plt.axis('off')
    augImage = batch[0]         
    augImage = augImage.astype('float32')
    augImage /= 255
    plt.imshow(augImage)
    i += 1

    if i > 19:

        break
		
for file in os.listdir(preprocessing_path):
    output = []
    img=Image.open(file)
    img=resize_aspect_ratio(img,(200,200))
    img = padding(img, (200, 200))
    output.append(np.array(img))
    image = np.squeeze(np.array(output))
    cv2.imwrite('p_{}'.format(file), image)
