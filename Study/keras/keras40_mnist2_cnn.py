random_seed=66

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# print(x_train[0])
# print(y_train[0])
# print(x_train[0].shape)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')/255. # dtype 도 float 로 바꿈
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/255.

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

model=Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3,3), padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(128, 3, padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

early=EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=early)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test[:10])

print(loss)
print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))

# results
# [0.06708189845085144, 0.9901999831199646]
# [[5.00446711e-18 4.33564296e-10 9.93366611e-10 4.09009465e-10
#   1.89408680e-11 3.10272801e-16 5.03198725e-21 1.00000000e+00
#   4.57957850e-14 5.88192517e-09]
#  [4.04008757e-31 2.50207736e-16 1.00000000e+00 1.18490236e-19
#   3.28232376e-25 0.00000000e+00 9.93651987e-34 9.00831110e-21
#   5.39170836e-26 1.01802090e-30]
#  [0.00000000e+00 1.00000000e+00 0.00000000e+00 0.00000000e+00
#   2.59818317e-35 0.00000000e+00 0.00000000e+00 1.07859656e-30
#   3.51940481e-25 0.00000000e+00]
#  [9.99907732e-01 2.97278036e-19 3.99415789e-09 2.05759411e-12
#   6.25348384e-11 5.79897641e-09 9.17348079e-05 1.74774041e-12
#   2.51535930e-08 4.87186867e-07]
#  [1.41384790e-27 6.87353485e-20 2.21151266e-18 7.39426854e-31
#   1.00000000e+00 7.16427256e-34 7.57093599e-19 2.09208879e-15
#   4.44357669e-21 6.58544920e-12]
#  [0.00000000e+00 1.00000000e+00 0.00000000e+00 0.00000000e+00
#   0.00000000e+00 0.00000000e+00 0.00000000e+00 2.21786030e-38
#   1.83208882e-31 0.00000000e+00]
#  [1.86101091e-12 4.69879025e-09 1.89829183e-08 6.85132832e-14
#   9.99984503e-01 3.12220092e-15 1.09269731e-08 4.75063871e-07
#   1.53632429e-09 1.49954521e-05]
#  [1.14029825e-16 8.43293085e-16 1.11023986e-15 2.63666016e-11
#   4.59512322e-08 8.97476848e-09 6.43536941e-14 1.80276960e-09
#   6.52770815e-09 1.00000000e+00]
#  [3.52719565e-09 7.85291192e-08 3.36056080e-13 1.09059918e-04
#   1.41761845e-11 9.99809444e-01 3.41242157e-05 8.79110743e-11
#   4.51396081e-05 2.16140506e-06]
#  [2.36967232e-21 3.28787104e-20 4.93712894e-20 2.25730452e-14
#   3.29802491e-10 4.12532543e-11 9.25056014e-18 4.93209466e-12
#   2.36977035e-11 1.00000000e+00]]
# [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# [7 2 1 0 4 1 4 9 5 9]