import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
# from tensorflow.keras import models
# from tensorflow import keras
from tensorflow.keras.layers import Dense

## 1. 데이터
x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([2,4,6,8,10,12,14,16,18,20])

x_test=np.array([101,102,103,104,105,106,107,108,109,110])
y_test=np.array([111,112,113,114,115,116,117,118,119,120])

x_predict=np.array([111,112,113])

## 2. 모델 구성
model=Sequential()
# model=models.Sequential()
# model=keras.models.Sequential()
model.add(Dense(110, input_dim=1, activation='relu'))
# relu = linear 보다 조금 더 성능이 좋은 함수 (평가지표 85% 이상)
model.add(Dense(1))

## 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10)

## 4. 평가 예측
loss=model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)

result=model.predict(x_predict)
print('result : ', result)