# fit 사용, numpy 저장

import tensorflow
import numpy as np

from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    BatchNormalization, Activation, Dense
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

# male = 841
# female = 895

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

train=datagen.flow_from_directory(
    './../data/image/data/',
    target_size=(128, 128),
    class_mode='binary',
    batch_size=27
)

for i in enumerate(range(5)):
    img, label=train.next()

x_train, x_test, y_train, y_test=train_test_split(img, label, train_size=0.8, random_state=23)

# test=datagen2.flow_from_directory(
#     './../data/image/data/',
#     target_size=(128, 128),
#     class_mode='binary',
#     batch_size=27
# )


model=Sequential()
model.add(Conv2D(128, 2, padding='same', input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.01), 
    metrics=['acc']
)

history=model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=5,
    validation_data=(x_test, y_test)
)

print('loss : ', history.history['loss'][-1])
print('acc : ', history.history['acc'][-1])
