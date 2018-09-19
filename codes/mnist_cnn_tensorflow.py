# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 07:08:42 2018

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

from data_package.mnist import MNIST
data=MNIST()
tf.reset_default_graph()
training_size=data.num_train
validation_size=data.num_val
test_size=data.num_test

image_size=data.img_size
image_size_flat=data.img_size_flat
image_shape_full=data.img_shape_full
num_channels=data.num_channels
num_classes=data.num_classes

filter_size1=5
num_filters1=16
filter_size2=5
num_filters2=36
fc_size=128



x_train=data.x_train
y_train=data.y_train
y_train_cls=data.y_train_cls


x=tf.placeholder(tf.float32,shape=[None,image_size_flat])
x_reshaped=tf.reshape(x,shape=[-1,image_size,image_size,num_channels])
y_true=tf.placeholder(tf.float32,shape=[None,num_classes])
y_true_cls=tf.argmax(y_true,axis=1)

w1=tf.Variable(tf.truncated_normal(shape=[filter_size1,filter_size1,num_channels,num_filters1],stddev=0.05))
b1=tf.Variable(tf.constant(0.05,shape=[num_filters1]))
conv_layer1=tf.nn.conv2d(x_reshaped,filter=w1,strides=[1,1,1,1],padding='SAME')
conv_layer1=conv_layer1+b1
conv_layer1=tf.nn.relu(conv_layer1)
pool_layer1=tf.nn.max_pool(conv_layer1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

w2=tf.Variable(tf.truncated_normal(shape=[filter_size2,filter_size2,num_channels,num_filters2],stddev=0.05))
b2=tf.Variable(tf.constant(0.05,shape=[num_filters2]))
conv_layer2=tf.nn.conv2d(x_reshaped,filter=w2,strides=[1,1,1,1],padding='SAME')
conv_layer2=conv_layer2+b2
conv_layer2=tf.nn.relu(conv_layer2)
pool_layer2=tf.nn.max_pool(conv_layer2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

tensorshape=pool_layer2.shape
num_features=tensorshape[1:4].num_elements()
flatten_layer=tf.reshape(pool_layer2,shape=[-1,num_features])

w3=tf.Variable(tf.truncated_normal(shape=[num_features,fc_size],stddev=0.05))
b3=tf.Variable(tf.constant(0.05,shape=[fc_size]))
fc1=tf.matmul(flatten_layer,w3)+b3

w4=tf.Variable(tf.truncated_normal(shape=[fc_size,num_classes],stddev=0.05))
b4=tf.Variable(tf.constant(0.05,shape=[num_classes]))
fc2=tf.matmul(fc1,w4)+b4

y_pred=tf.nn.softmax(fc2)
y_pred_cls=tf.argmax(y_pred,axis=1)

cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_cls,logits=fc2)
cost=tf.reduce_mean(cross_entropy)
optimizer=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction=tf.equal(y_pred_cls,y_true_cls)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))


total_iteration=100
train_batch_size=100



export_dir="model/MNIST/tensorflow/"

if not os.path.exists(export_dir):
    os.makedirs(export_dir)

saver=tf.train.Saver()
writer=tf.summary.FileWriter("model/MNIST/cifar/1")

with tf.Session() as session:
    
    
    session.run(tf.global_variables_initializer())

    for i in range(total_iteration):
        x_batch,y_batch,y_batch_cls=data.random_batch(batch_size=train_batch_size)
        feed_dict_train={x:x_batch,y_true:y_batch}
        session.run(optimizer,feed_dict=feed_dict_train)
        if i%10==0:
            acc=session.run(accuracy,feed_dict=feed_dict_train)
            msg='Iteration:{0},accuracy:{1:%}'
            print(msg.format(i+1,acc))
    
    writer.add_graph(session.graph)
    
#    model_dir=os.path.join(export_dir,'test_model')
#    model_graph_dir=os.path.join(export_dir,'test_model.meta')
#    saver.save(session,os.path.join(export_dir,'test_model'))















