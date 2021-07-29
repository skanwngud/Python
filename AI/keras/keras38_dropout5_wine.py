# DNN 

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
from tensorflow.keras.layers import Dense, LSTM, Input, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
y=y.reshape(-1, 1)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=78)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=78)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

enc=OneHotEncoder()
enc.fit(y_train)
y_train=enc.transform(y_train).toarray()
y_test=enc.transform(y_test).toarray()
y_val=enc.transform(y_val).toarray()

input1=Input(shape=(13,))
dense1=Dense(250, activation='relu')(input1)
dense1=Dropout(0.2)(dense1)
dense1=Dense(250, activation='relu')(dense1)
dense1=Dense(250, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
dense1=Dense(60, activation='relu')(dense1)
dense1=Dense(60, activation='relu')(dense1)
dense1=Dense(30, activation='relu')(dense1)
dense1=Dense(30, activation='relu')(dense1)
output1=Dense(3, activation='softmax')(dense1)
model=Model(input1, output1)

model.compile(loss='mse', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test[:5])

print(loss)
print(y_pred)
print(np.argmax(y_pred, axis=-1))

# [0.003051705891266465, 1.0]
# [[9.9965537e-01 3.4037273e-04 4.3318109e-06]
#  [9.9875712e-01 1.2073033e-03 3.5540870e-05]
#  [5.2045932e-04 9.9932301e-01 1.5649901e-04]
#  [2.6215613e-04 9.9971694e-01 2.0802659e-05]
#  [1.0816221e-10 1.2158861e-04 9.9987841e-01]]
# [0 0 1 1 2]

# dropout
# [0.0003076987632084638, 1.0]
# [[9.9875522e-01 1.2447579e-03 2.5181655e-11]
#  [9.9882871e-01 1.1713360e-03 2.0326189e-11]
#  [1.0713656e-03 9.9801564e-01 9.1301597e-04]
#  [2.5687678e-04 9.9958652e-01 1.5659045e-04]
#  [1.1816288e-06 2.7776996e-03 9.9722111e-01]]
# [0 0 1 1 2]