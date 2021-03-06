import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
# from tensorflow.keras import models
# from tensorflow import keras
from tensorflow.keras.layers import Dense

## 1. 데이터
x_train=np.array([1,2,3,4,5])
y_train=np.array([1,2,3,4,5])

x_validation=np.array([6,7,8])
y_validation=np.array([6,7,8])

x_test=np.array([9,10,11])
y_test=np.array([9,10,11])
# 사실상 train, validation, test 를 분리 시켜놨지만 전부 하나의 데이터이다
# 실제 데이터 준비 시 하나의 데이터를 잘 나눠서 배치
# 총 4 종류의 데이터, train, test, validation, predict (predict 는 y의 값이 필요없다)

## 2. 모델 구성
model=Sequential()
# model=models.Sequential()
# model=keras.models.Sequential()
model.add(Dense(1000, input_dim=1, activation='linear'))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

## 3. 컴파일, 훈련
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_validation, y_validation)
)

## 4. 평가 예측
loss=model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)

#result=model.predict([9])
result=model.predict(x_train)
print('result : ', result)