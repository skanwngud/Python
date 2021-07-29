# m31 로 만든 0.95 이상의 n_component 를 이용하여 dnn 만들 것

import numpy as np

from tensorflow.keras.datasets import mnist

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. data
(x_train, y_train), (x_test, y_test)=mnist.load_data()

print(x_train.shape) # (60000, 28, 28)
print(x_test.shape) # (10000, 28, 28)
print(y_train.shape) # (60000, 28, 28)
print(y_test.shape) # (10000, 28, 28)

x_train=x_train.reshape(-1, 28*28)/255
x_test=x_test.reshape(-1, 28*28)/255

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

pca=PCA(154)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)

# 2. model
model=Sequential()
model.add(Dense(64, activation='relu', input_shape=(154,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. compile
es=EarlyStopping(patience=10)
rl=ReduceLROnPlateau(verbos=1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=2, validation_split=0.2, callbacks=[es, rl])

# 4. evaluate
loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print(loss)
print(np.argmax(y_pred[:10], axis=-1))
print(np.argmax(y_test[:10], axis=-1))


# [0.11212018132209778, 0.9794999957084656]
# [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
# [7 2 1 0 4]

# [0.12745237350463867, 0.9746000170707703]
# [7 2 1 0 4 1 4 9 5 9]
# [7 2 1 0 4 1 4 9 5 9]