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

data_generator=datagen.flow_from_directory(
    './../data/image/data/male',
    target_size=(128, 128),
    class_mode='binary',
    batch_size=179
)

# print(data_generator[0][0])

# female_generator=datagen2.flow_from_directory(
#     'c:/data/image/data/female',
#     target_size=(128, 128),
#     class_mode='binary',
#     batch_size=32
# )


# train, test=train_test_split(
#     data_generator,
#     train_size=0.8,
#     random_state=32
# )

# print(train[0])

# model=Sequential()
# model.add(Conv2D(128, 2, padding='same', input_shape=(128, 128, 3)))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))

# model.summary()

# model.compile(
#     loss='binary_crossentropy',
#     optimizer=Adam(learning_rate=0.1),
#     metrics=['acc']
# )

# model.fit(
#     train[0][0],
#     train[0][1],
#     steps_per_epoch=10,
#     validation_steps=4,
#     validation_data=test,
#     epochs=3
# )

# loss=model.evaluate(
#     test
# )

# print('loss : ', loss[0])
# print('acc : ', loss[1])
