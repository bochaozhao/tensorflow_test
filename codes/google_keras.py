# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 17:45:26 2018

@author: bchao
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train=pd.read_csv('data/stock/Google_Stock_Price_Train.csv')
training_set=dataset_train.iloc[:,1:2].values

total_length=len(dataset_train)
sequence_length=60
num_units=50
keep_prob=0.8
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
x_train = []
y_train = []
for i in range(sequence_length, total_length):
    x_train.append(training_set_scaled[i-sequence_length:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))

training_size=len(x_train)





# Part 2 - Building the RNN
# Importing the Keras libraries and packages
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import LSTM, Dense, Flatten, Dropout
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

inputs=Input(shape=(sequence_length,1,))
net=inputs
net=LSTM(units=num_units,return_sequences=True,dropout=0.0,recurrent_dropout=0.0)(net)
net=LSTM(units=num_units,return_sequences=True,dropout=0.0,recurrent_dropout=0.0)(net)
net=LSTM(units=num_units,return_sequences=True,dropout=0.0,recurrent_dropout=0.0)(net)
net=LSTM(units=num_units,return_sequences=False,dropout=0.0,recurrent_dropout=0.0)(net)
net=Dense(units=1)(net)  #note that no activation function is used
outputs=net
model=Model(inputs=inputs,outputs=outputs)
optimizer=Adam(lr=1e-4)
model.compile(optimizer=optimizer,loss='mean_squared_error')


log_dir='model/Google_stock/test/'
callback_log=TensorBoard(log_dir=log_dir,histogram_freq=0,batch_size=32,write_graph=True,
                             write_grads=False,write_images=False)

model.fit(x_train,y_train,epochs=100,batch_size=32,callbacks=[callback_log])






# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('data/stock/Google_Stock_Price_Test.csv')
y_test = dataset_test.iloc[:, 1:2].values

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
y_pred = model.predict(x_test)
y_pred = sc.inverse_transform(y_pred)

# Visualising the results
plt.plot(y_test, color = 'red', label = 'Real Google Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
