# DNN 구성
# keras23 보다 loss 더 좋게 만들 것

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x=np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred=np.array([50,60,70])

x_pred=x_pred.reshape(1,3)

print(x.shape)
print(x_pred.shape)

model=Sequential()
model.add(Dense(300, activation='relu', input_shape=(3,)))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(1))

early=EarlyStopping(monitor='loss', patience=200, mode='auto')
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, callbacks=early)

loss=model.evaluate(x,y)
y_pred=model.predict(x_pred)

print(loss)
print(y_pred)

# results
# 2.7238973416388035e-05
# [[79.99382]]