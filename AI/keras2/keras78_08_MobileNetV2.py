# 실습 - cifar10 으로 MobileNetV2 넣어서 만들 것

import tensorflow
import numpy as np

from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Activation
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.datasets import cifar10

from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test)=cifar10.load_data()

x_train, x_val, y_train, y_val=train_test_split(
    x_train, y_train,
    train_size=0.8,
    random_state=32
)

print(x_train.shape)

x_train=x_train/255.
x_val=x_val/255.
x_test=x_test/255.

MobileNetV2=MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

MobileNetV2.trainable=False

model=Sequential()
model.add(MobileNetV2)
model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(10, activation='softmax'))

es=EarlyStopping(
    monitor='val_loss',
    patience=50,
    verbose=1
)

rl=ReduceLROnPlateau(
    monitor='val_loss',
    patience=30,
    verbose=1
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics='acc'
)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[es, rl]
)

loss=model.evaluate(
    x_test, y_test
)

y_pred=model.predict(
    x_test
)

print(loss)
print(np.argmax(y_pred[:5], axis=-1))
print(np.argmax(y_test[:5], axis=-1))

# [3.9525020122528076, 0.30169999599456787]
# [3 1 0 8 6]