from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
from numpy import array
# 기존에 쓰던 np.array()를 array() 로 쓸 수가 있음

#1. data
x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([1,2,3,4,5,6,7,8,9,10])

x_test=array([11,12,13,14,15])
y_test=array([11,12,13,14,15])

x_pred=array([16,17,18])

#2. modeling
model=Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. compille, training
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)
# validation_split=0.2 : model.fit 에 들어가는 변수들의 데이터를 20%를 분리해서 validation data 로 바꿔 훈련함

#4. evaluate, predict
results=model.evaluate(x_test, y_test, batch_size=1)
# result 에 들어가는 변수 : loss, metrics
print("results : ", results)

y_pred=model.predict(x_pred)
# predict 는 y 값을 구하는 것이므로 y 데이터로 y_pred 를 넣을 필요가 없음
print('y_pred : ', y_pred)

