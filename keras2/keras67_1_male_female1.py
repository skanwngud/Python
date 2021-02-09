# fit_generator 사용

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
    rescale=1./255,
    validation_split=0.2
)

datagen2=ImageDataGenerator()

male=datagen.flow_from_directory(
    'c:/data/image/data/',
    target_size=(128, 128),
    class_mode='binary',
    batch_size=32,
    subset="training"

)

male2=datagen.flow_from_directory(
    'c:/data/image/data/',
    target_size=(128, 128),
    class_mode='binary',
    batch_size=32,
    subset="validation"

)

print(male)
print(male2)

# female=datagen.flow_from_directory(
#     'c:/data/image/data/',
#     target_size=(128, 128),
#     class_mode='binary',
#     batch_size=32
# )

# print(male[0][0].shpae)
# for i in enumerate(range(5)):
#     img, label=data_generator.next()

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

model=Sequential()
model.add(Conv2D(128, 2, padding='same', input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.1),
    metrics=['acc']
)

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
es=EarlyStopping(patience=20, verbose=1, monitor='loss')
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='loss')

history = model.fit_generator(
    male,
    steps_per_epoch=10,
    epochs=100,
    callbacks=[es, rl],
    validation_data=male2
)

print('loss : ', history.history['loss'][-1])
print('acc : ', history.history['acc'][-1])

# results
# Epoch 00048: early stopping
# loss :  0.6074425578117371
# acc :  0.659375011920929
