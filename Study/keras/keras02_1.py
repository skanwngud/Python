# 네이밍 룰

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
# from tensorflow.keras import models
# from tensorflow import keras
from tensorflow.keras.layers import Dense

## 1. 데이터
x_train=np.array([1,2,3,4,5])
y_train=np.array([1,2,3,4,5])
# 데이터가 많으면 많을수록 학습이 더 잘 됨
# 훈련용 데이터

x_test=np.array([6,7,8])
y_test=np.array([6,7,8])
# 평가용 데이터

## 2. 모델 구성
model=Sequential()
# model=models.Sequential()
# model=keras.models.Sequential()
# 경로 지정을 해주면 임포트를 거치지 않아도 된다
model.add(Dense(1000, input_dim=1, activation='linear'))
# w와 b를 연산하는 과정 중에 activation이라는 것이 존재 (일단 개념만 알아둘 것)
model.add(Dense(1000))
# 아무것도 적지 않은 경우에는 default 값으로 정의
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
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

## 4. 평가 예측
loss=model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)

result=model.predict([9])
print('result : ', result)