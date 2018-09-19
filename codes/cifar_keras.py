# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 22:47:44 2018

@author: bchao
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
from data_package.cifar10 import CIFAR10
data=CIFAR10()


#from tensorflow.python.keras import backend as K
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import InputLayer, Input
#from tensorflow.python.keras.layers import Reshape, MaxPooling2D
#from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
#from tensorflow.python.keras.callbacks import TensorBoard
#from tensorflow.python.keras.optimizers import Adam
#from tensorflow.python.keras.models import load_model
#from tensorflow.python.keras.models import Model




tf.reset_default_graph()

training_size=data._num_images_train

test_size=10000

image_size=data.img_size
image_size_flat=data.img_size_flat

num_channels=data.num_channels
num_classes=data.num_classes
name_classes=data.name
image_shape=(image_size,image_size)
image_shape_full=(image_size,image_size,num_channels)


x_train=data.x_train

y_train=data.y_train
y_train_cls=data.y_train_cls
x_test=data.x_test
y_test=data.y_test
y_test_cls=data.y_test_cls


img_size_cropped=24
filter_size1=5
num_filters1=16
filter_size2=5
num_filters2=36
fc_size=128
keep_probability=0.8
train_batch_size=100
test_batch_size=256





from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
training_set=train_datagen.flow(x=x_train,y=y_train,batch_size=32)



inputs=Input(shape=(image_size,image_size,num_channels))
net=inputs
net=Conv2D(kernel_size=filter_size1,strides=1,filters=num_filters1,padding='same',activation='relu',name='layer_conv1')(net)
net=MaxPooling2D(pool_size=2,strides=2)(net)
net=Conv2D(kernel_size=filter_size2,strides=1,filters=num_filters2,padding='same',activation='relu',name='layer_conv2')(net)
net=MaxPooling2D(pool_size=2,strides=2)(net)
net=Flatten()(net)
net=Dropout(rate=1-keep_probability)(net)
net=Dense(fc_size,activation='relu')(net)
net=Dense(num_classes,activation='softmax')(net)
outputs=net

model=Model(inputs=inputs,outputs=outputs)
optimizer=Adam(lr=1e-4)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])





model.fit_generator(training_set,steps_per_epoch=100,epochs=2)


#model.fit(x=x_train,y=y_train,epochs=2,batch_size=train_batch_size,validation_split=0.2)


export_dir="model/CIFAR-10/keras/"

if not os.path.exists(export_dir):
    os.makedirs(export_dir)

model_dir="model/CIFAR-10/keras/model.keras"
model.save(model_dir)

model_load=load_model(model_dir)

model_load.summary()

test_loss,test_acc=model_load.evaluate(x=x_test,y=y_test)

















