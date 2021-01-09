# dnn model - (N, 28, 28) -> (N, 764) = (N, 28*28)

import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=55)

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])/255.
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])/255.
x_val=x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2])/255.

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

print(x_train.shape) # 37632000, 1
print(y_train.shape) # 48000,

# print(x_train[0])

model=Sequential()
model.add(Dense(200, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

early=EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val), callbacks=early)

loss=model.evaluate(x_test, y_test)
pred=model.predict(x_test[:5])

print(loss)

print(y_test[:5])
print(np.argmax(pred, axis=-1))

# results

# [0.11212018132209778, 0.9794999957084656]
# [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
# [7 2 1 0 4]