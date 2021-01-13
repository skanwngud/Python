import numpy as np

x=np.load('../data/npy/diabets_x.npy')
y=np.load('../data/npy/diabets_y.npy')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

x=x.reshape(x.shape[0], x.shape[1], 1)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=11)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=11)

input=Input(shape=(x.shape[1], 1)) # (10, 1)
cnn1=Conv1D(100, 2, padding='same')(input)
max1=MaxPooling1D(2)(cnn1)
drop1=Dropout(0.2)(max1)
cnn1=Conv1D(100, 2, padding='same')(drop1)
max1=MaxPooling1D(2)(cnn1)
drop1=Dropout(0.2)(max1)
dense1=Dense(100, activation='relu')(drop1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
output=Dense(1)(dense1)
model=Model(input, output)

# input=Input(shape=(x.shape[1], 1))
# lstm1=LSTM(100, activation='relu')(input)
# dense1=Dense(100, activation='relu')(lstm1)
# dense1=Dense(120, activation='relu')(dense1)
# dense1=Dense(150, activation='relu')(dense1)
# dense1=Dense(120, activation='relu')(dense1)
# dense1=Dense(100, activation='relu')(dense1)
# dense1=Dense(80, activation='relu')(dense1)
# output=Dense(1,)(dense1)
# model=Model(input, output)

model.summary()
'''
es=EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), callbacks=es)

loss=model.evaluate(x_test, y_test)

print(loss)

# results - conv1D
# 4758.693359375

# results - lstm (126/200)
# 3929.65771484375
'''