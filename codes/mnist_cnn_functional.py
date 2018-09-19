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


from data_package.mnist import MNIST
data=MNIST()

training_size=data.num_train
validation_size=data.num_val
test_size=data.num_test

image_size=data.img_size
image_size_flat=data.img_size_flat
image_shape=data.img_shape
image_shape_full=data.img_shape_full
num_channels=data.num_channels
num_classes=data.num_classes
x_train=data.x_train
y_train=data.y_train
y_train_cls=data.y_train_cls
x_test=data.x_test
y_test=data.y_test
y_test_cls=data.y_test_cls

filter_size1=5
num_filters1=16
filter_size2=5
num_filters2=36
fc_size=128
keep_probability=0.8
train_batch_size=100
test_batch_size=256




x=tf.placeholder(tf.float32,shape=[None,image_size_flat])
keep_prob=tf.placeholder(tf.float32)
x_reshaped=tf.reshape(x,shape=[-1,image_size,image_size,num_channels])
y_true=tf.placeholder(tf.float32,shape=[None,num_classes])
y_true_cls=tf.argmax(y_true,axis=1)

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05,shape=[length]))

def new_conv_layer(input,num_input_channels,filter_size,num_filters,strides=[1,1,1,1],use_pooling=True):
    shape=[filter_size,filter_size,num_input_channels,num_filters]
    weights=new_weights(shape=shape)
    biases=new_biases(length=num_filters)
    layer=tf.nn.conv2d(input,filter=weights,strides=strides,padding='SAME')
    layer+=biases
    layer=tf.nn.relu(layer)
    
    return layer,weights

def new_pooling_layer(input,ksize,strides):
    layer=tf.nn.max_pool(input,ksize=ksize,strides=strides,padding='SAME')
    return layer

def new_dropout_layer(input,keep_prob):
    layer=tf.nn.dropout(input,keep_prob=keep_prob)
    return layer

def new_flatten_layer(input):
    tensorshape=input.shape
    num_features=tensorshape[1:4].num_elements()
    layer=tf.reshape(input,shape=[-1,num_features])
    return layer, num_features

def new_fc_layer(input,num_inputs,num_outputs,use_relu=True,use_sigmoid=False):
    weights=new_weights(shape=[num_inputs,num_outputs])
    biases=new_biases(length=num_outputs)
    layer=tf.matmul(input,weights)+biases
    if use_relu:
        layer=tf.nn.relu(layer)
    if use_sigmoid:
        layer=tf.nn.sigmoid(layer)
    return layer


def plot_images(images,cls_true,cls_pred=None):
    assert len(images)==len(cls_true)==9
    
    fig,axes=plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    
    for i,ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(image_shape),cmap='binary')
        if cls_pred is None:
            xlabel="True:{0}".format(cls_true[i])
        else:
            xlabel="True:{0},Pred:{1}".format(cls_true[i],cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    


def plot_example_errors(cls_pred,correct,starting_index=0):
    incorrect=(correct==False)
    images=data.x_test[incorrect]
    cls_pred=cls_pred[incorrect]
    cls_true=data.y_test_cls[incorrect]
    plot_images(images=images[starting_index:(starting_index+9)],cls_true=cls_true[starting_index:(starting_index+9)]
            ,cls_pred=cls_pred[starting_index:(starting_index+9)])


def get_training_accuracy():
    x_batch,y_true_batch,y_batch_cls=data.random_batch(batch_size=train_batch_size)
    feed_dict_train={x:x_batch,y_true:y_true_batch,keep_prob:keep_probability}
    acc=session.run(accuracy,feed_dict=feed_dict_train)
    return acc

def get_testing_accuracy(show_example_errors=False):
    cls_pred=np.zeros(shape=test_size,dtype=np.int)
    i=0
    while i<test_size:
        j=min(i+test_batch_size,test_size)
        images=data.x_test[i:j,:]
        labels=data.y_test[i:j,:]
        feed_dict_test={x:images,y_true:labels,keep_prob:1.0}
        cls_pred[i:j]=session.run(y_pred_cls,feed_dict=feed_dict_test)
        i=j
    cls_true=data.y_test_cls
    correct=(cls_true==cls_pred)
    correct_sum=correct.sum()
    acc=float(correct_sum)/test_size
      
    if show_example_errors:
        plot_example_errors(cls_pred,correct,starting_index=0)
        
    return acc


def plot_conv_weights(weights,input_channel=0):
    w=session.run(weights)
    w_min=np.min(w)
    w_max=np.max(w)
    num_filters=w.shape[3]
    num_grids=math.ceil(math.sqrt(num_filters))
    fig,axes=plt.subplots(num_grids,num_grids)
    for i,ax in enumerate(axes.flat):
        if i<num_filters:
            img=w[:,:,input_channel,i]
            im=ax.imshow(img,vmin=w_min,vmax=w_max,interpolation='nearest',cmap='seismic')
            
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(right=0.8)
    cbar_ax=fig.add_axes([0.85,0.15,0.05,0.7])
    fig.colorbar(im,cax=cbar_ax)
    plt.show()

def plot_conv_layer(layer,image):
    feed_dict={x:[image]}
    values=session.run(layer,feed_dict=feed_dict)
    num_filters=values.shape[3]
    num_grids=math.ceil(math.sqrt(num_filters))
    fig,axes=plt.subplots(num_grids,num_grids)
    for i,ax in enumerate(axes.flat):
        if i<num_filters:
            img=values[0,:,:,i]
            im=ax.imshow(img,interpolation='nearest',cmap='seismic')
            
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(right=0.8)
    cbar_ax=fig.add_axes([0.85,0.15,0.05,0.7])
    fig.colorbar(im,cax=cbar_ax)
    plt.show()


    

def optimize(num_iterations,interval):
    iteration=np.zeros(math.ceil(num_iterations/interval))
    training_acc=np.zeros(math.ceil(num_iterations/interval))
    testing_acc=np.zeros(math.ceil(num_iterations/interval))
    j=0
    for i in range(num_iterations):
        x_batch,y_true_batch,y_batch_cls=data.random_batch(batch_size=train_batch_size)
        feed_dict_train={x:x_batch,y_true:y_true_batch,keep_prob:keep_probability}
        start_time=time.time()
        session.run(optimizer,feed_dict=feed_dict_train)
        end_time=time.time()
        optimize_time=end_time-start_time
        if i%interval==0:
            acc_train=get_training_accuracy()
            start_test_time=time.time()
            acc_test=get_testing_accuracy()
            end_test_time=time.time()
            test_time=end_test_time-start_test_time
            if i==0:
                msg='optimize time per iteration:{0:>6},testing time:{1:>6}'
                print(msg.format(optimize_time,test_time))
            msg='Iteration:{0:>6},training accuracy:{1:>6.1%},testing accuracy:{2:>6.1%}'
            print(msg.format(i+1,acc_train,acc_test))
            iteration[j]=i
            training_acc[j]=acc_train
            testing_acc[j]=acc_test
            j+=1
    return iteration,training_acc,testing_acc


conv_layer1,weights1=new_conv_layer(x_reshaped,num_channels,filter_size1,num_filters1,strides=[1,1,1,1])
pool_layer1=new_pooling_layer(conv_layer1,ksize=[1,2,2,1],strides=[1,2,2,1])
conv_layer2,weights2=new_conv_layer(pool_layer1,num_filters1,filter_size2,num_filters2,strides=[1,1,1,1])
pool_layer2=new_pooling_layer(conv_layer2,ksize=[1,2,2,1],strides=[1,2,2,1])
flatten_layer,flatten_size=new_flatten_layer(pool_layer2)
fc_layer1=new_fc_layer(flatten_layer,flatten_size,fc_size,use_relu=True)
dropout_layer=new_dropout_layer(fc_layer1,keep_prob=keep_prob)
fc_layer2=new_fc_layer(dropout_layer,fc_size,num_classes,use_relu=True)


y_pred=tf.nn.softmax(fc_layer2)
y_pred_cls=tf.argmax(y_pred,axis=1)

cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_cls,logits=fc_layer2)
cost=tf.reduce_mean(cross_entropy)
optimizer=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction=tf.equal(y_pred_cls,y_true_cls)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))




session=tf.Session()
session.run(tf.global_variables_initializer())

iteration,training_acc,testing_acc=optimize(num_iterations=100,interval=10)

accuracy=get_testing_accuracy(show_example_errors=True)

session.close()










