import numpy as np

x=np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred=np.array([50,60,70])
# y_pred = 80

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv1D, LSTM, MaxPooling1D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8)

print(x_train.shape) # (10, 3)
print(y_train.shape) # (10, )

x_train=x_train.reshape(10, 3, 1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_pred.shape) # (3, )
x_pred=x_pred.reshape(1, 3, 1)

# input=Input(shape=(x_train.shape[1], 1))
# cnn1=Conv1D(10, 2, padding='valid')(input)
# max1=MaxPooling1D(2)(cnn1)
# drop1=Dropout(0.2)(max1)
# dense1=Dense(100, activation='relu')(drop1)
# dense1=Dense(120, activation='relu')(dense1)
# dense1=Dense(150, activation='relu')(dense1)
# dense1=Dense(120, activation='relu')(dense1)
# dense1=Dense(100, activation='relu')(dense1)
# dense1=Dense(80, activation='relu')(dense1)
# output=Dense(1)(dense1)
# model=Model(input, output)

input=Input(shape=(x_train.shape[1], 1))
lstm1=LSTM(10, activation='relu')(input)
dense1=Dense(100, activation='relu')(lstm1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
output=Dense(1)(dense1)
model=Model(input, output)

es=EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, callbacks=es)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_pred)

print(loss)
print(y_pred)

# results - conv1d
# 26.72039222717285
# [[[75.41567]]]

# results - lstm
# 7.504388332366943
# [[89.38371]]