import numpy as np
import tensorflow as tf
## 1. 데이터 준비
x = np.array([1,2,3])
y = np.array([1,2,3])

## 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# 순차적으로 연산하는 모델을 생성
model.add(Dense(5, input_dim=1, activation='linear'))
# Dense : y=wx+b에 근간을 둔 모델, 5 : node 갯수, input_dim=1 : input이 하나
model.add(Dense(3, activation='linear'))
# activation='linear' : 활성화 함수(선형 함수를 사용)
model.add(Dense(4))
# model.add : model 에 layer를 쌓음
model.add(Dense(1))

## 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs = 1000, batch_size=1)
# batch_size : 전체 데이터를 몇 번으로 나누어 훈련할지
# epochs : 데이터 전체의 훈련을 몇 번 반복할 지
# model.compile : 컴파일함, model.fit : 모델을 훈련시킴

## 4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
print('loss :', loss)
# 위의 모델을 평가함

result = model.predict([4])
print('result :', result)
# 학습 된 모델을 통해 값을 예측함