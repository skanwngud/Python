random_seed=66

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')/255.
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/255.

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

model.save('../data/h5/k52_1_model1.h5')

modelpath='../data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
early=EarlyStopping(monitor='val_loss', patience=5, mode='auto')
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
hist=model.fit(x_train, y_train, epochs=30, batch_size=64, validation_split=0.2, callbacks=[early, cp])

model.save('../data/h5/k52_1_model2.h5')
model.save_weights('../data/h5/k52_1_weight.h5', )

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print(loss)
print(np.argmax(y_pred[:10], axis=-1))
print(np.argmax(y_test[:10], axis=-1))

# results
# [0.05556643009185791, 0.9904000163078308]
# [7 2 1 0 4 1 4 9 5 9]
# [7 2 1 0 4 1 4 9 5 9]