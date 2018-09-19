# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:03:38 2018

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

tf.reset_default_graph()
from data_package.mnist import MNIST
data=MNIST()

training_size=data.num_train
validation_size=data.num_val
test_size=data.num_test

image_size=data.img_size
image_size_flat=data.img_size_flat
image_shape_full=data.img_shape_full
image_shape=data.img_shape
num_channels=data.num_channels
num_classes=data.num_classes

dis_size=128
latent_size=100
gen_size=128
filter_size1=5
num_filters1=16
filter_size2=5
num_filters2=36
fc_size=128

gen_dense_size=7
generator_num_filters1=36
generator_filter_size1=5
generator_num_filters2=16
generator_filter_size2=5

generator_filter_size3=5

x_train=data.x_train
y_train=data.y_train
y_train_cls=data.y_train_cls


def plot_images(images):
    assert len(images)==9
    
    fig,axes=plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    
    for i,ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(image_shape),cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def leakyrelu(x):
    return tf.maximum(x,tf.multiply(x,0.2))


#discriminator net
def discriminator(images,reuse=None):
    activation=leakyrelu
    with tf.variable_scope("discriminator",reuse=reuse):
        x_reshaped=tf.reshape(images,shape=[-1,image_size,image_size,num_channels])
        D_conv_layer1=tf.layers.conv2d(x_reshaped,kernel_size=filter_size1,filters=num_filters1,strides=2,padding='same',activation=activation)
        D_conv_layer2=tf.layers.conv2d(D_conv_layer1,kernel_size=filter_size2,filters=num_filters2,strides=2,padding='same',activation=activation)
        D_flatten=tf.contrib.layers.flatten(D_conv_layer2)
        D_output=tf.layers.dense(D_flatten,units=1,activation=tf.nn.sigmoid)
        
        return D_output


#x=tf.placeholder(tf.float32,shape=[None,image_size_flat],name='x')
#x_reshaped=
def generator(g,is_training):
    activation=leakyrelu
    with tf.variable_scope("generator",reuse=None):
        G_dense=tf.layers.dense(g,units=gen_dense_size*gen_dense_size,activation=activation)
        G_dense=tf.contrib.layers.batch_norm(G_dense,is_training=is_training)
        G_dense_reshaped=tf.reshape(G_dense,shape=[-1,gen_dense_size,gen_dense_size,1])
        G_dense_resized=tf.image.resize_images(G_dense_reshaped,size=[7,7])
        G_deconv_layer1=tf.layers.conv2d_transpose(G_dense_resized,filters=generator_num_filters1,kernel_size=generator_filter_size1,
                                           strides=2,padding='SAME',activation=activation)
        G_deconv_layer1=tf.contrib.layers.batch_norm(G_deconv_layer1,is_training=is_training)
        G_deconv_layer2=tf.layers.conv2d_transpose(G_deconv_layer1,filters=generator_num_filters2,kernel_size=generator_filter_size2,
                                           strides=2,padding='SAME',activation=activation)
        G_deconv_layer2=tf.contrib.layers.batch_norm(G_deconv_layer2,is_training=is_training)
        G_output=tf.layers.conv2d_transpose(G_deconv_layer2,filters=num_channels,kernel_size=generator_filter_size3,strides=1,padding='SAME',activation=tf.nn.sigmoid)
        
        return G_output,G_deconv_layer1


#generator net
g=tf.placeholder(tf.float32,shape=[None,latent_size],name='g')
is_training=tf.placeholder(tf.bool,name='is_training')
G_output,G_deconv_layer1=generator(g,is_training)

x=tf.placeholder(tf.float32,shape=[None,image_size_flat],name='x')
D_real=discriminator(x)
D_fake=discriminator(G_output,reuse=True)



eps=1e-12
D_loss=-tf.reduce_mean(tf.log(D_real+eps)+tf.log(1.0-D_fake+eps))
G_loss=-tf.reduce_mean(tf.log(D_fake+eps))

[grad_G_g,grad_G_layer1]=tf.gradients(G_loss,xs=[g,G_deconv_layer1])
[grad_D_x,grad_D_fake]=tf.gradients(D_loss,xs=[x,G_output])
grad_G_g=tf.reduce_mean(grad_G_g)
grad_G_layer1=tf.reduce_mean(grad_G_layer1)
grad_D_x=tf.reduce_mean(grad_D_x)
grad_D_fake=tf.reduce_mean(grad_D_fake)
summary_grad_G_g=tf.summary.scalar(name='gradient_G_input',tensor=grad_G_g)
summary_grad_G_layer1=tf.summary.scalar(name='gradient_G_layer1',tensor=grad_G_layer1)
summary_grad_D_x=tf.summary.scalar(name='gradient_D_true',tensor=grad_D_x)
summary_grad_D_fake=tf.summary.scalar(name='gradient_D_fake',tensor=grad_D_fake)
merged_summary_G=tf.summary.merge([summary_grad_G_g,summary_grad_G_layer1])
merged_summary_D=tf.summary.merge([summary_grad_D_x,summary_grad_D_fake])


vars_G=[var for var in tf.trainable_variables() if var.name.startswith("generator")]
vars_D=[var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):  
    D_optimizer=tf.train.AdamOptimizer(learning_rate=1e-3).minimize(D_loss,name='D_optimizer',var_list=vars_D)
    G_optimizer=tf.train.AdamOptimizer(learning_rate=1e-3).minimize(G_loss,name='G_optimizer',var_list=vars_G)

def G_sample(batch_size,latent_size):
    return np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])

def real_sample(batch_size):
    # Create a random index into the training-set.
    idx = np.random.randint(low=0, high=training_size, size=batch_size)

    # Use the index to lookup random training-data.
    x_batch = x_train[idx,:]

    return x_batch
    
num_iteration=20000
num_iteration_pretrain=1
batch_size=32


export_dir="model/DCGAN/MNIST/test/"

if not os.path.exists(export_dir):
    os.makedirs(export_dir)

saver=tf.train.Saver()
writer=tf.summary.FileWriter("model/DCGAN/MNIST/test/")


with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    
    #############################################################################################################
    #pretrain the discriminator
    for i in range(num_iteration_pretrain):        
        print("pretraining:{0}".format(i+1))
        x_batch=real_sample(batch_size=batch_size)
        x_generated=G_sample(batch_size=batch_size,latent_size=latent_size)
            
        feed_dict_D={x:x_batch,g:x_generated,is_training:True}
        _, D_loss_train,summary_pretrain=session.run([D_optimizer,D_loss,merged_summary_D],feed_dict=feed_dict_D)
#        writer.add_summary(summary_pretrain,global_step=i+1)
    #adversarail training
    
    D_loss_train=10
    G_loss_train=10
    num_train_D=0
    num_train_G=0
    for i in range(num_iteration):
        train_D=True
        train_G=True

        if D_loss_train*2<G_loss_train:
            train_D=False
        if G_loss_train*1.5<D_loss_train:
            train_G=False
        
        if train_D:        
            x_batch=real_sample(batch_size=batch_size)
            x_generated=G_sample(batch_size=batch_size,latent_size=latent_size)
            
            feed_dict_D={x:x_batch,g:x_generated,is_training:True}
            _, D_loss_train,summary_D=session.run([D_optimizer,D_loss,merged_summary_D],feed_dict=feed_dict_D)
            
            num_train_D+=1
#            writer.add_summary(summary_D,global_step=num_train_D)
        if train_G:
            
            x_generated=G_sample(batch_size=batch_size,latent_size=latent_size)
            feed_dict_G={g:x_generated,is_training:True}
            _, G_generated,G_loss_train,summary_G=session.run([G_optimizer,G_output,G_loss,merged_summary_G],feed_dict=feed_dict_G)
            num_train_G+=1
#            writer.add_summary(summary_G,global_step=num_train_G)
        if i%10==0:
            msg='G trained:{0},D_trained:{1}'
            print(msg.format(num_train_G,num_train_D))
            plot_images(G_generated[0:9])
            
#    writer.add_graph(session.graph)
#    model_dir=os.path.join(export_dir,'test_model')
#    model_graph_dir=os.path.join(export_dir,'test_model.meta')
#    saver.save(session,os.path.join(export_dir,'test_model'))
        
        