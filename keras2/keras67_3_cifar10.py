# fit_generator, flow

import tensorflow
import numpy as np

from keras.datasets import cifar10

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout,\
    Dense, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical


datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=0.1,
    vertical_flip=True,
    horizontal_flip=True,
    zoom_range=0.2,
    rescale=1./255
)

datagen2=ImageDataGenerator()

(x_train, y_train), (x_test, y_test)=cifar10.load_data()

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

trainset=datagen.flow(
    x_train, y_train,
    batch_size=500
)

testset=datagen2.flow(
    x_test, y_test,
    batch_size=500
)

# print(x_train.shape) # (50000, 32, 32, 3)
# print(y_train.shape) # (50000, 1)

model=Sequential()
model.add(Conv2D(64, 2, padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.1),
    metrics=['acc']
)

es=EarlyStopping(patience=30, verbose=1)
rl=ReduceLROnPlateau(patience=15, verbose=1)

history=model.fit_generator(
    trainset,
    epochs=5,
    steps_per_epoch=100,
    validation_data=testset,
    validation_steps=5,
    callbacks=[es, rl]
)

acc=model.evaluate_generator(testset)

print('loss : ', acc[0])
print('acc : ', acc[1])

# results
# loss :  1606.736572265625
# acc :  0.0997999981045723

'''
# data
(x_train, y_train), (x_test, y_test)=cifar10.load_data()

x_train=x_train.reshape(-1, 32, 32, 3)/255.
x_test=x_test.reshape(-1, 32, 32, 3)/255.

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

# model
model=Sequential()
model.add(Conv2D(64, 2, padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

es=EarlyStopping(
    patience=30,
    verbose=1
)

rl=ReduceLROnPlateau(
    patience=15,
    verbose=1
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.1),
    metrics=['acc'])
history=model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.2,
    callbacks=[es, rl]
)
'''
print(history)

# 과제
# ImageDataGenerator 에서 데이터가 증폭 되었다는 증거

# (model.summary?, matplotlib?, history)

# Epoch 5/5
# 1250/1250 [==============================] - 5s 4ms/step - loss: 1.5094 - acc: 0.4275 - val_loss: 1.5570 - val_acc: 0.4416
# <tensorflow.python.keras.callbacks.History object at 0x000001CF339A0D90>

