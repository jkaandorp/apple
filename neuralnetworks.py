# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:19:09 2018

@author: jimka
"""

import numpy as np
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\jimka\Dropbox\Programmeren\Spyder\S&P500ticdata\daily_data_analysis.csv',index_col = 'time')

#n = data.shape[0]*2//3
#train_data = data.iloc[:n]   
#test_data = data.iloc[n:]   


# regression statsmodels
#Y_train = train_data['ln_r_d']
#X_train = train_data[['AR(1)','AR(2)','AR(3)','vola_t-1','vola_t-2','vola_t-3', 'skew_t-1']]
#X_test = test_data[['AR(1)','AR(2)','AR(3)','vola_t-1','vola_t-2','vola_t-3', 'skew_t-1']]
#y_test = test_data['ln_r_d']


# RNN neural network
#
#model = Sequential()
#model.add(LSTM(128, imput_shape=(X_train.shape[1:]), return_sequences = True))
#model.add(Dropout(0,2))
#model.add(Batchnormalization())
#
#model = Sequential()
#model.add(LSTM(128, imput_shape=(X_train.shape[1:]), return_sequences = True))
#model.add(Dropout(0.1))
#model.add(Batchnormalization())
#
#model = Sequential()
#model.add(LSTM(128, imput_shape=(X_train.shape[1:])))
#model.add(Dropout(0.2))
#model.add(Batchnormalization())

model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation = 'softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(Loss='sparce_categorical_crossentropy', optimizer = opt, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir= f'logs/{NAME}')

X[t, features]
X[t - start: t, features, t]

X_train.shape
window_size = 260
X_train_extended = np.zeros((X_train.shape[0] - windiw_size, window_size, 5))
y_train_extended = np.zeros(X_train.shape[0] - window_size)
for i in range((X_train.shape[0] - window_size):
    X_train_extended[i, :, :] = X_train.loc[X_train.index[i : i + window_size], :]
    y_train_extended[i] = y_train[i + window_size]
X_train_extended[1, :, :]

X_test.shape
X_test_extended = np.zeros((X_test.shape[0] - window_size, window_size, 5))
y_test_extended = np.zeros(X_test.shape[0] - window_size)
for i in range(X_test.shape[0] - window_size):
    X_test_extended[i, :, :] = X_test.loc[X_test.index[i : i + window_size], :]
    y_test_extended[i] = y_test[i + window_size]
X_test_extended[1, :, :]


# RNN

X_train.shape[1]
model = Sequential()
model.add(LSTM(input_shape=(X_train_extended.shape[1], X_train_extended.shape[2]), units=100))
model.add(Dropout(0.3))
model.add(Dense(2*X_train.shape[1]))
model.add(Dense(1))

model.compile(optimizer=SGD(lr=0.001), loss='mse')
history = model.fit(X_train_extended, y_train_extended, validation_data=(X_test_extended,
#                                                                         y_test_extended), epochs=50)
for i in range(1250):
    model.train_on_batch(np.reshape(X_train_extended[i], (1, X_train_extended.shape[1], 
                                    X_train_extended.shape[2])), np.reshape([y_train_extended[i]], (1, 1)))
    
model.summary()
plt.plot(history.history['loss'])

len(X_train)


# simple neural network

X_train.shape[1]
model = Sequential()
model.add(Dense(input_shape=(X_train.shape[1], ), units=100))
model.add(Dropout(0.3))
model.add(Dense(2*X_train.shape[1]))
model.add(Dense(1))

model.compile(optimizer=SGD(lr=0.001), loss='mse')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)


model.summary()
plt.plot(history.history['loss'])



y_pred = model.predict(X_test)
y_pred = y_pred.flatten()
test_data['prediction'] = y_pred
test_data.head()

test_data.drop(['AR(1)','AR(2)','AR(3)','vola_t-1','vola_t-2','vola_t-3', 'skew_t-1', 'vola', 'skew', 'price'], axis=1, inplace=True)    

test_data['outperformance'] = test_data['ln_r_d'] - test_data['prediction']

exp_returns = np.exp(test_data)
cum_exp_returns = exp_returns.cumprod()

cum_exp_returns.plot()

cum_outperformance = test_data.cumsum()
cum_outperformance_exp = np.exp(cum_outperformance)

cum_outperformance.head()


cum_outperformance_exp.head()
