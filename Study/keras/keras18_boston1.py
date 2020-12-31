import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

dataset=load_boston()
x=dataset.data
y=dataset.target
# x, y로 분류 (x = 집값에 영향을 끼치는 요소, y = 집값)

# print(x.shape) # (506, 13)
# print(y.shape) # (506, )
# print('='*20)
# print(x[:5])
# print(y[:10])

# # 데이터를 소수로 치환하여 계산하면 0 ~ 1 사이에 들어가므로 연산의 부담이 적어짐
# # 위의 데이터는 전처리가 다 되어있는지는 불명 (어느 정도 다듬어지기는 했음)

# print(np.max(x), np.min(x)) # 711.0 0.0
# print(dataset.feature_names) # column name
# print(dataset.DESCR) # dataset describe

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)

# model=Sequential()
# model.add(Dense(10, input_dim=13, activation='relu'))
# model.add(Dense(15))
# model.add(Dense(1))

# 모델 구성
input1=Input(shape=13)
dense1=Dense(35, activation='relu')(input1)
dense1=Dense(35, activation='relu')(dense1)
dense1=Dense(35, activation='relu')(dense1)
dense1=Dense(35, activation='relu')(dense1)
dense1=Dense(35, activation='relu')(dense1)
dense1=Dense(35, activation='relu')(dense1)
output1=Dense(1)(dense1)
model=Model(inputs=input1, outputs=output1)

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=130, batch_size=1, validation_split=0.2)

# 평가, 예측
loss=model.evaluate(x_test, y_test, batch_size=1)
pred=model.predict(x_test)

def RMSE(y_test, pred):
    return np.sqrt(mean_squared_error(y_test, pred))

rmse=RMSE(y_test, pred)
r2=r2_score(y_test, pred)

print(loss)
print(rmse)
print(r2)

# 결과
# [20.876935958862305, 3.1320741176605225]
# 4.5691282708228105
# 0.7502249403803791