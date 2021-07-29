# (-1, 32, 32, 3) -> Flatten -> Reshape -> (-1, 32, 32, 3)

import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape

(x_train, y_train), (x_test, y_test)=cifar10.load_data()

print(x_train.shape)
print(x_test.shape)

y_train=x_train
y_test=x_test

model=Sequential()
model.add(Dense(64, activation='relu', input_shape=(32,32,3)))
model.add(Dense(128, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(32*32*3))
model.add(Reshape((32,32,3)))
model.add(Dense(3))

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train,
            epochs=2, batch_size=128)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print(y_pred[0])
print(y_pred.shape)