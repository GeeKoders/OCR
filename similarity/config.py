import os

IMG_SHAPE = (200, 200)
NUM_CLASSES = 2
BATCH_SIZE = 128
EPOCHS = 50
TRAINING_PATH = os.path.join(os.getcwd(), 'training-data')
BASE_OUTPUT = 'output'
MODEL_NAME = 'CNN_Siamese.h5'
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, 'model'])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, 'plot.png'])