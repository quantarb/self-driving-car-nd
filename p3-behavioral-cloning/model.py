
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, Convolution2D, Lambda, Activation, MaxPooling2D, Cropping2D
from keras.layers.advanced_activations import ELU
from keras.preprocessing.image import ImageDataGenerator 
from keras.optimizers import SGD
from keras.layers import Input
import os

data_dirname = "data"
csv = "driving_log.csv"

central_images = driving_log["center"]
steering_angles = driving_log["steering"]

n_train = len(driving_log)

X_train =[]
y_train = []


for index, image_path in enumerate(central_images):
    f = os.path.join(data_dirname, image_path)
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    steering = steering_angles[index]
    X_train.append(img)
    y_train.append(steering)
    
    rimg = cv2.flip(img, 1)
    r_steering = steering * -1
    X_train.append(rimg)
    y_train.append(r_steering)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train, y_train = shuffle(X_train, y_train)

def resize(image):
    import tensorflow as tf
    return tf.image.resize_images(image, (16, 32))

def grayscale(image):
    import tensorflow as tf
    return tf.image.rgb_to_grayscale(image)

def normalize(image):
    '''Normalize the image to be between -0.5 and 0.5'''
    return image / 255.0 - 0.5

def base_model(image_dims):
    ch, row, col = image_dims 
    model = Sequential()
    model.add(Lambda(resize, input_shape=(ch, row, col)))
    model.add(Lambda(grayscale))
    model.add(Lambda(normalize))    
    return model

def simple_model(image_dims):
    model = base_model(image_dims)
    
    model.add(Convolution2D(32, 3, 3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(43))
    model.add(Activation('relu'))
    model.add(Dense(1))        
    model.compile(optimizer="adam", loss="mse")
    
    return model

model = simple_model(image_dims)

batch_size = 50
nb_epoch = 5

model.fit(X_train, y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_split=0.1)


model.save_weights("model.h5")
print('saving weights')

# Save model config (architecture)
json_string = model.to_json()
with open("model.json", "w") as f:
    f.write(json_string)

print('saving model')