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

x_train=np.load('.././data/image/data/train_set.npy')
y_train=np.load('.././data/image/data/test_set.npy')
x_val=np.load('.././data/image/data/val_set.npy')
y_val=np.load('.././data/image/data/val_test_set.npy')

x_train, x_test, y_train, y_test=train_test_split(
    x_train,
    y_train,
    train_size=0.8,
    random_state=23
)

# datagen=ImageDataGenerator(
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.2
# )

# datagen2=ImageDataGenerator()

# train=datagen.flow(
#     x_train,
#     y_train,
#     batch_size=16
# )

# val=datagen2.flow(
#     x_val,
#     y_val,
#     batch_size=16
# )

model=Sequential()
model.add(Conv2D(128, 2, padding='same', input_shape=(64, 64, 3)))
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
    optimizer=Adam(learning_rate=1.0, epsilon=None), 
    metrics=['acc']
)

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

es=EarlyStopping(
    monitor='val_loss',
    patience=30,
    verbose=1
)

rl=ReduceLROnPlateau(
    monitor='val_loss',
    patience=10,
    verbose=1
)

history=model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    batch_size=16,
    epochs=1000,
    callbacks=[es, rl]
)

print('loss : ', history.history['loss'][-1])
print('acc : ', history.history['acc'][-1])

# results
# Epoch 00031: early stopping
# loss :  0.0
# acc :  0.4932493269443512

# Epoch 00031: early stopping
# loss :  0.0
# acc :  0.48424842953681946