# 23_3 copy, LSTM 2 layers

import numpy as np

# 1. data
x=np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred=np.array([50,60,70])

# 2. modeling
# x=x.reshape(13,3,1)
x=x.reshape(x.shape[0], x.shape[1], 1) # x.shape[0] = 13, x.shape[1] = 1

x_pred=x_pred.reshape(1,3,1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

input1=Input(shape=(3, 1))
# dense1=LSTM(330, activation='relu')(input1)
# dense1=Dense(330, activation='relu')(dense1)
# dense1=Dense(330, activation='relu')(dense1)
# dense1=Dense(330, activation='relu')(dense1)
# output1=Dense(1)(dense1)
dense1=LSTM(330, activation='relu', return_sequences=True)(input1)
dense1=LSTM(330, activation='relu', return_sequences=True)(dense1)
# 상위 레이어의 아웃풋 노드는 하위 레이어의 인풋
# LSTM 2개를 엮는 것 말고는 return_sequences 는 잘 안 쓰임
dense1=LSTM(330, activation='relu')(dense1)
dense1=Dense(330, activation='relu')(dense1)
output1=Dense(1)(dense1)
model=Model(inputs=input1, outputs=output1)
# LSTM 을 여러층 쌓을 땐 retrun_sequences=True 를 사용한다
# LSTM 2 layers 인 경우 첫 번째 layer 에서 나온 값은 시계열 데이터가 아니기 때문에 2 layer 부터 값이 안 좋게 나온다
# (1 layer 에서 나오는 output 이 시계열 데이터라 판단할 수 있으면 LSTM 을 두개 이상 쌓을 수 있다)
model.summary()
# return_sequences=True 인 경우 은닉층의 연산값을 전부 출력하기 때문에?

'''
# 3. compile, fitting
early=EarlyStopping(monitor='loss', patience=50, mode='auto')
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1, callbacks=early)

# 4. evaluate, predict
loss=model.evaluate(x, y)
y_pred=model.predict(x_pred)

print(loss)
print(y_pred)

# result - LSTM 1 layer
# 0.07782885432243347
# [[80.32742]]

# result - LSTM 2 layers
# 0.14005020260810852
# [[80.499466]]

# result - LSTM 3 layers
# 0.5436415672302246
# [[79.509445]]
'''