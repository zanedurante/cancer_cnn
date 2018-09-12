#various imports
import os
import pandas as pd
import numpy as np
import PIL
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from PIL import Image, ImageFile
from sklearn.decomposition import IncrementalPCA
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from cancer_detection.preproccessing import preproccess_scripts

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#change paths when on server
IMAGE_DATA_PATH = '/hdd/datasets/melanoma_data/Images/'
DESC_DATA_PATH = '/hdd/datasets/melanoma_data/Descriptions/'

RESIZE_WIDTH = 224
RESIZE_HEIGHT = 224

#load in dataset into dataframe
df, images = preproccess_scripts.load_data(IMAGE_DATA_PATH, DESC_DATA_PATH, start=0, end = 14000)

#function to convert malignant to 1 and benign to 0

#go through dataframe
#function to convert malignant to 1 and benign to 0

#create empty list
labels = []

#go through dataframe
for index, row in df.iterrows():
    if row['benign_malignant'] == 'malignant':
        labels.append(1)
    else :
        labels.append(0)
    

#convert to numpy array
values = np.array(labels)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, values, test_size = .20)


#initialize the model
rs50 = ResNet50(include_top=False, weights='imagenet', input_shape = (244,244,3))

for layer in rs50.layers:
    layer.trainable=False

top = Sequential()
top.add(rs50)
top.add(Flatten())
top.add(Dense(256,activation='relu'))
top.add(Dropout(0.3))
top.add(Dense(256,activation='relu'))
top.add(Dropout(0.3))
top.add(Dense(1,activation='sigmoid'))

#compile
top.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#change number of epochs
for i in range(20):
    top.fit(x_train, y_train, epochs=1, shuffle=True)
    score  = top.evaluate(x_test, y_test)
    top.save("weights2/weights_acc"+str(score)+".h5py")

