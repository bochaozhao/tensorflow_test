# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 16:20:17 2018

@author: bchao
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Model


import data_package.imdb as imdb

imdb.maybe_download_and_extract()

x_train_text,y_train=imdb.load_data(train=True)
x_test_text, y_test = imdb.load_data(train=False)

print("Train-set size: ", len(x_train_text))
print("Test-set size:  ", len(x_test_text))

data_text = x_train_text + x_test_text

x_train_text[1]
y_train[1]


num_words = 10000
tokenizer = Tokenizer(num_words=num_words)



tokenizer.fit_on_texts(data_text)

tokenizer.word_index

x_train_tokens = tokenizer.texts_to_sequences(x_train_text)

np.array(x_train_tokens[1])

x_test_tokens = tokenizer.texts_to_sequences(x_test_text)

num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

np.mean(num_tokens)
np.max(num_tokens)
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
max_tokens

np.sum(num_tokens < max_tokens) / len(num_tokens)

pad = 'pre'
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)


x_train_pad.shape
x_test_pad.shape

idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))

x_train=x_train_pad
x_test=x_test_pad
sequence_length=max_tokens

def tokens_to_string(tokens):
    # Map from tokens back to words.
    words = [inverse_map[token] for token in tokens if token != 0]
    
    # Concatenate all words.
    text = " ".join(words)

    return text

#x_train_text[0]
#tokens_to_string(x_train_pad[0])

######################################################################################################################################################


embedding_size=8
num_units=16



inputs=Input(shape=(sequence_length,))
net=inputs
net=Embedding(input_dim=num_words,output_dim=embedding_size,input_length=sequence_length,name='layer_embedding')(net)
net=GRU(units=num_units,return_sequences=True)(net)
net=GRU(units=num_units,return_sequences=True)(net)
net=GRU(units=num_units)(net)
net=Dense(units=1,activation='sigmoid')(net)
outputs=net
model=Model(inputs=inputs, outputs=outputs)

optimizer=Adam(lr=1e-4)
model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

model.summary()
log_dir='model/imdb_sentinent/GRU/'
callback_log=TensorBoard(log_dir=log_dir,histogram_freq=0,batch_size=32,write_graph=True,
                             write_grads=False,write_images=False)

model.fit(x_train,y_train,validation_split=0.05,epochs=3,batch_size=32,callbacks=[callback_log])


test_result = model.evaluate(x_test, y_test)

print("Accuracy: {0:.2%}".format(test_result[1]))

################################################################################################################################################



y_pred = model.predict(x=x_test[0:1000])

cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in y_pred])
cls_true = np.array(y_test[0:1000])

incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]

idx = incorrect[0]
idx
text = x_test_text[idx]
text
y_pred[idx]


##################################################################################################################################

text1 = "This movie is fantastic! I really like it because it is so good!"
text2 = "Good movie!"
text3 = "Maybe I like this movie."
text4 = "Meh ..."
text5 = "If I were a drunk teenager then this movie might be good."
text6 = "Bad movie!"
text7 = "Not a good movie!"
text8 = "This movie really sucks! Can I get my money back please?"
texts = [text1, text2, text3, text4, text5, text6, text7, text8]


tokens = tokenizer.texts_to_sequences(texts)

tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)
tokens_pad.shape

model.predict(tokens_pad)


##################################################################################################################################

layer_embedding = model.get_layer('layer_embedding')
weights_embedding = layer_embedding.get_weights()[0]


token_good = tokenizer.word_index['good']

token_great = tokenizer.word_index['great']
weights_embedding[token_good]

weights_embedding[token_great]

































