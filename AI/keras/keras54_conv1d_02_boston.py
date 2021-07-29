import numpy as np

from tensorflow.keras.layers import Dense, Input, Conv1D, MaxPooling1D, Dropout, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

x=np.load('../data/npy/boston_x.npy')
y=np.load('../data/npy/boston_y.npy')

scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

x=x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=22)
x_test, x_val, y_test, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=22)

# input=Input(shape=(x.shape[1], 1))
# cnn1=Conv1D(100, 2, padding='same')(input)
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


input=Input(shape=(x.shape[1], 1))
lstm1=LSTM(100, activation='relu')(input)
dense1=Dense(100, activation='relu')(lstm1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
dense1=Dense(60, activation='relu')(dense1)
output=Dense(1)(dense1)
model=Model(input, output)

es=EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), callbacks=es)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print(loss)

# result - conv1D
# 51.99382781982422

# result - lstm
# 15.361844062805176