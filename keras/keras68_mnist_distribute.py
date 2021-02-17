# 분산처리

random_seed=66

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')/255. # dtype 도 float 로 바꿈
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/255.

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
early=EarlyStopping(monitor='loss', patience=5, mode='auto')



strategy=tf.distribute.MirroredStrategy(cross_device_ops=\
    tf.distribute.HierarchicalCopyAllReduce()
    )
# 하이퍼 파라미터가 다섯개 정도 있지만 Hier~ 가 제일 잘 먹힘

with strategy.scope():
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

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=early)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test[:10])

print(loss)
print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))
