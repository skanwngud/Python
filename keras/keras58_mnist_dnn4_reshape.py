random_seed=66

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')/255.
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/255.

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# y_train=to_categorical(y_train)
# y_test=to_categorical(y_test)

y_train=x_train
y_test=x_test

print(y_train.shape) # (600000, 28, 28, 1)
print(y_test.shape) # (600000, 28, 28, 1)


model=Sequential()
model.add(Dense(64, input_shape=(28, 28, 1)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(784, activation='relu'))
model.add(Reshape((28, 28, 1))) # Reshape 할 레이어 위에는 항상 Reshape 할만큼의 아웃풋 레이어가 있어야함
model.add(Dense(1))

model.summary()

modelpath='..\data\modelcheckpoint\k57_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'

early=EarlyStopping(monitor='val_loss', patience=10, mode='auto')
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
reduce_lr=ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1,)

model.compile(loss='mse', optimizer='adam', metrics='acc')
hist=model.fit(x_train, y_train, epochs=3, batch_size=512, validation_split=0.5, callbacks=[early, cp, reduce_lr])

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print(loss)
print(y_pred[0])
print(y_pred.shape)
