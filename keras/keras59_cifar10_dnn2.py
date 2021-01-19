# (-1, 32, 32, 3) -> (-1, 10)

import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test)=cifar10.load_data()

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

print(x_train.shape) # (50000, 32, 32, 3)
print(y_train.shape) # (10000, 32, 32, 3)

model=Sequential()
model.add(Dense(128, input_shape=(32, 32, 3)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10))

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train,
            epochs=100, batch_size=64)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print(y_pred.shape)