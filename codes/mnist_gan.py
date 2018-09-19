# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 11:57:45 2018

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
    


#discriminator net
x=tf.placeholder(tf.float32,shape=[None,image_size_flat],name='x')
D_w1=tf.Variable(tf.truncated_normal(shape=[image_size_flat,dis_size],stddev=0.05),name='D_w1')
D_b1=tf.Variable(tf.constant(0.05,shape=[dis_size]),name='D_b1')

D_w2=tf.Variable(tf.truncated_normal(shape=[dis_size,1],stddev=0.05),name='D_w2')
D_b2=tf.Variable(tf.constant(0.05,shape=[1]),name='D_b2')

var_D=[D_w1,D_b1,D_w2,D_b2]





#generator net
g=tf.placeholder(tf.float32,shape=[None,latent_size],name='g')
G_w1=tf.Variable(tf.truncated_normal(shape=[latent_size,gen_size],stddev=0.05),name='G_w1')
G_b1=tf.Variable(tf.constant(0.05,shape=[gen_size]),name='G_b1')

G_w2=tf.Variable(tf.truncated_normal(shape=[gen_size,image_size_flat],stddev=0.05),name='G_w2')
G_b2=tf.Variable(tf.constant(0.05,shape=[image_size_flat]),name='G_b2')
G_layer1=tf.nn.relu(tf.matmul(g,G_w1)+G_b1)
G_logit=tf.matmul(G_layer1,G_w2)+G_b2
G_fake=tf.nn.sigmoid(G_logit)
var_G=[G_w1,G_b1,G_w2,G_b2]

D_real_layer1=tf.nn.relu(tf.matmul(x,D_w1)+D_b1)
D_real_logit=tf.matmul(D_real_layer1,D_w2)+D_b2
D_real=tf.nn.sigmoid(D_real_logit)

D_fake_layer1=tf.nn.relu(tf.matmul(G_fake,D_w1)+D_b1)
D_fake_logit=tf.matmul(D_fake_layer1,D_w2)+D_b2
D_fake=tf.nn.sigmoid(D_fake_logit)



D_loss=-tf.reduce_mean(tf.log(D_real)+tf.log(1.0-D_fake))
G_loss=-tf.reduce_mean(tf.log(D_fake))

[grad_G_g,grad_G_layer1]=tf.gradients(G_loss,xs=[g,G_layer1])
[grad_D_x,grad_D_fake]=tf.gradients(D_loss,xs=[x,G_fake])
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



D_optimizer=tf.train.AdamOptimizer(learning_rate=1e-3).minimize(D_loss,name='D_optimizer',var_list=var_D)
G_optimizer=tf.train.AdamOptimizer(learning_rate=1e-3).minimize(G_loss,name='G_optimizer',var_list=var_G)

def G_sample(batch_size,latent_size):
    return np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])

def real_sample(batch_size):
    # Create a random index into the training-set.
    idx = np.random.randint(low=0, high=training_size, size=batch_size)

    # Use the index to lookup random training-data.
    x_batch = x_train[idx,:]

    return x_batch
    
num_iteration=1000
num_iteration_discriminator=1
num_iteration_pretrain=100
batch_size=32


export_dir="model/GAN/MNIST/test/"

if not os.path.exists(export_dir):
    os.makedirs(export_dir)

saver=tf.train.Saver()
writer=tf.summary.FileWriter("model/GAN/MNIST/test/")


with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    
    #############################################################################################################
    #pretrain the discriminator
    for i in range(num_iteration_pretrain):        
        x_batch=real_sample(batch_size=batch_size)
        x_generated=G_sample(batch_size=batch_size,latent_size=latent_size)
            
        feed_dict_D={x:x_batch,g:x_generated}
        _, D_loss_train,summary_pretrain=session.run([D_optimizer,D_loss,merged_summary_D],feed_dict=feed_dict_D)
        writer.add_summary(summary_pretrain,global_step=i+1)
    #adversarail training
    for i in range(num_iteration):
        for k in range(num_iteration_discriminator):        
            x_batch=real_sample(batch_size=batch_size)
            x_generated=G_sample(batch_size=batch_size,latent_size=latent_size)
            
            feed_dict_D={x:x_batch,g:x_generated}
            _, D_loss_train,summary_D=session.run([D_optimizer,D_loss,merged_summary_D],feed_dict=feed_dict_D)
            writer.add_summary(summary_D,global_step=num_iteration_pretrain+i+1)
                    
        x_generated=G_sample(batch_size=batch_size,latent_size=latent_size)
        feed_dict_G={g:x_generated}
        _, G_generated,G_loss_train,summary_G=session.run([G_optimizer,G_fake,G_loss,merged_summary_G],feed_dict=feed_dict_G)
        writer.add_summary(summary_G,global_step=i+1)
        if i%100==0:
            msg='Iteration:{0},G_loss:{1:%}'
            print(msg.format(i+1,G_loss_train))
            plot_images(G_generated[0:9])
            
    writer.add_graph(session.graph)
#    model_dir=os.path.join(export_dir,'test_model')
#    model_graph_dir=os.path.join(export_dir,'test_model.meta')
#    saver.save(session,os.path.join(export_dir,'test_model'))
        
        
       
       
       
        
    
    
    



























