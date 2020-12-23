import numpy as np
import tensorflow as tf

## 1. 데이터 준비
x = np.array([1,2,3])
y = np.array([1,2,3])
# 데이터의 수치만을 보고 데이터의 정제됨을 판단하면 안 됨
#(e.g. y=[1,2,3000000]은 혼자 수치가 크지만 정상적인 데이터 값)

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
# batch_size : 전체 데이터를 몇 번으로 나누어 훈련할 지
# epochs : 데이터 전체의 훈련을 몇 번 반복할 지
# model.compile : 컴파일함, model.fit : 모델을 훈련시킴

## 4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
print('loss :', loss)
# 위의 모델을 평가함
# 트레인 데이터로 학습하고 트레인 데이터로 평가하여 신뢰도가 떨어짐 (테스트 데이터로 평가해야함)
x_pred = np.array([4])
result = model.predict(x_pred)

#result = model.predict([4])
print('result :', result)
# 학습 된 모델을 통해 값을 예측함