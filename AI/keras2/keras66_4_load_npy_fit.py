import numpy as np

# numpy 불러오기
x_train=np.load('c:/data/image/brain/npy/keras66_train_x.npy')
y_train=np.load('c:/data/image/brain/npy/keras66_train_y.npy')
x_test=np.load('c:/data/image/brain/npy/keras66_test_x.npy')
y_test=np.load('c:/data/image/brain/npy/keras66_test_y.npy')

# print(x_train.shape, y_train.shape) # (160, 150, 150, 3) (160, )
# print(x_test.shape, y_test.shape) # (120, 150, 150, 3) (120, )

# 모델 만들기 실습

import tensorflow

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, \
    BatchNormalization, Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# model
model=Sequential()
model.add(Conv2D(32, 3, padding='same', input_shape=(150, 150, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Conv2D(64, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Conv2D(128, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))

es=EarlyStopping(patience=100, verbose=1)
rl=ReduceLROnPlateau(patience=20, verbose=1)

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.1),
                metrics=['acc'])
model.fit(x_train, y_train, epochs=1000, batch_size=32,
            validation_split=0.2, callbacks=[es, rl])

loss=model.evaluate(x_test, y_test)

print('loss : ', loss[0])
print('acc : ', loss[1])

# loss :  1.4540479183197021
# acc :  0.800000011920929