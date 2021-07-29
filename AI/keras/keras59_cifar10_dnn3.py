# (-1, 32, 32, 3) - > (-1, 32, 32, 3)

import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

(x_train, y_train), (x_test, y_test)=cifar10.load_data()

y_train=x_train
y_test=x_test

print(x_train.shape) # (50000, 32, 32, 3)
print(y_train.shape) # (10000, 32, 32, 3)

model=Sequential()
model.add(Dense(128, input_shape=(32, 32, 3)))
# model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x_train, y_train,
            epochs=100, batch_size=64)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print(y_pred.shape)
