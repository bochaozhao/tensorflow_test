# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 11:11:36 2018

@author: bchao
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
dataset_train=pd.read_csv('data/stock/Google_Stock_Price_Train.csv')
training_set=dataset_train.iloc[:,1:2].values

# Getting the real stock price of 2017
dataset_test = pd.read_csv('data/stock/Google_Stock_Price_Test.csv')
y_test = dataset_test.iloc[:, 1:2].values


tf.reset_default_graph()

total_length=len(dataset_train)
sequence_length=60
num_units=50
keep_prob=0.8
batch_size=32
total_iteration=2000
keep_prob=0.8
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
y_test=sc.transform(y_test)
# Creating a data structure with 60 timesteps and 1 output
x_train = []
y_train = []
for i in range(sequence_length, total_length):
    x_train.append(training_set_scaled[i-sequence_length:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
y_train=np.reshape(y_train,(y_train.shape[0],1))
training_size=len(x_train)




# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - sequence_length:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test = []
for i in range(sequence_length, len(inputs)):
    x_test.append(inputs[i-sequence_length:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))




def random_batch(batch_size=32):
    """
    Create a random batch of training-data.

    :param batch_size: Number of images in the batch.
    :return: 3 numpy arrays (x, y, y_cls)
    """

    # Create a random index into the training-set.
    idx = np.random.randint(low=0, high=training_size, size=batch_size)

        # Use the index to lookup random training-data.
    x_batch = x_train[idx,:,:]
    y_batch = y_train[idx]

    return x_batch, y_batch

lstm_sizes=[num_units,num_units,num_units]
x=tf.placeholder(tf.float32,shape=[None,sequence_length,1])
y_true=tf.placeholder(tf.float32,shape=[None,1])
lstm_cells=[tf.nn.rnn_cell.BasicLSTMCell(num_units=size) for size in lstm_sizes] 
drops=[tf.contrib.rnn.DropoutWrapper(cell=lstm_cell,output_keep_prob=keep_prob) for lstm_cell in lstm_cells]
lstm_stacked=tf.nn.rnn_cell.MultiRNNCell(cells=drops)
#initial_state=lstm_stacked.zero_state(batch_size=batch_size,dtype=tf.float32)
lstm_outputs,final_state=tf.nn.dynamic_rnn(cell=lstm_stacked,inputs=x,dtype=tf.float32)
y_pred=tf.contrib.layers.fully_connected(inputs=lstm_outputs[:,-1,:],num_outputs=1,activation_fn=None)
loss=tf.losses.mean_squared_error(labels=y_true,predictions=y_pred)
optimizer=tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(loss)

export_dir="model/Google_stock/tensorflow/"

if not os.path.exists(export_dir):
    os.makedirs(export_dir)

saver=tf.train.Saver()
writer=tf.summary.FileWriter("model/Google_stock/tensorflow/")

with tf.Session() as session:
    
    
    session.run(tf.global_variables_initializer())

    for i in range(total_iteration):
        x_batch,y_true_batch=random_batch(batch_size=batch_size)
        feed_dict_train={x:x_batch,y_true:y_true_batch}
        session.run(optimizer,feed_dict=feed_dict_train)
        if i%10==0:
            train_loss=session.run(loss,feed_dict=feed_dict_train)
            feed_dict_test={x:x_test,y_true:y_test}
            test_loss=session.run(loss,feed_dict=feed_dict_test)
            msg='Iteration:{0},train_loss:{1:.1%},test_loss:{2:.1%}'
            print(msg.format(i+1,train_loss,test_loss))

    feed_dict_test={x:x_test,y_true:y_test}
    y_predicted=session.run(y_pred,feed_dict=feed_dict_test)
    feed_dict_train={x:x_train,y_true:y_train}
    y_train_predicted=session.run(y_pred,feed_dict=feed_dict_train)
    
    
    writer.add_graph(session.graph)
    
    model_dir=os.path.join(export_dir,'test_model')
    model_graph_dir=os.path.join(export_dir,'test_model.meta')
    saver.save(session,os.path.join(export_dir,'test_model'))
    
    
# Visualising the results
real_price=sc.inverse_transform(y_test)
predicted_price=sc.inverse_transform(y_predicted)
plt.plot(real_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()    
    
# Visualising the results
train_real_price=sc.inverse_transform(y_train)
train_predicted_price=sc.inverse_transform(y_train_predicted)
plt.plot(train_real_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(train_predicted_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    