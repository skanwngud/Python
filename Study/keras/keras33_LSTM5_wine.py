# LSTM

from sklearn.datasets import load_wine

dataset=load_wine()

x=dataset.data
y=dataset.target

# print(x)
# print(y)
# print(x.shape) # (178, 13)
# print(y.shape) # (178, )

import numpy as np

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, SimpleRNN, GRU
from tensorflow.keras.callbacks import EarlyStopping
y=y.reshape(-1, 1)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=78)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=78)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

x_trian=x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_vap=x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

enc=OneHotEncoder()
enc.fit(y_train)
y_train=enc.transform(y_train).toarray()
y_test=enc.transform(y_test).toarray()
y_val=enc.transform(y_val).toarray()

input1=Input(shape=(13,1))
lstm1=LSTM(250, activation='relu')(input1)
dense1=Dense(250, activation='relu')(lstm1)
dense1=Dense(250, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
dense1=Dense(60, activation='relu')(dense1)
dense1=Dense(60, activation='relu')(dense1)
dense1=Dense(50, activation='relu')(dense1)
dense1=Dense(50, activation='relu')(dense1)
dense1=Dense(30, activation='relu')(dense1)
dense1=Dense(30, activation='relu')(dense1)
output1=Dense(3, activation='softmax')(dense1)
model=Model(input1, output1)

early=EarlyStopping(monitor='acc', patience=20, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=150, validation_data=(x_val, y_val))

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test[:5])

print(loss)
print(y_pred)
print(np.argmax(y_pred, axis=-1))

# [0.032978605479002, 0.9444444179534912]
# [[9.6455628e-01 3.5441197e-02 2.4996955e-06]
#  [8.4969288e-01 1.4968589e-01 6.2124548e-04]
#  [1.5620500e-03 9.9374020e-01 4.6977992e-03]
#  [2.6861643e-03 9.9285549e-01 4.4584172e-03]
#  [4.9879523e-06 3.8159948e-02 9.6183509e-01]]
# [0 0 1 1 2]