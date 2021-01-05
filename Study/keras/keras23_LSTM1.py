# 1. data
import numpy as np

x=np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y=np.array([4,5,6,7])

# print('x.shape : ', x.shape) # (4, 3)
# print('y.shape : ', y.shape) # (4, )

x=x.reshape(4,3,1)

# 2. modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model=Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3, 1)))
# LSTM 을 사용하기 위해서 3차원 데이터여야한다 (Dnse 는 2차원)
# input_shape=(3,1)에서 3 = timesteps, 1 = input_dim
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 10)                480 - 4 * ((몇 개씩 쪼갤건지 + 1(bias)) * 노드 + 노드^2 = 4*(1+1)*10+10^2
_________________________________________________________________
dense (Dense)                (None, 20)                220         4 * (input_dim + bias + output) * output
_________________________________________________________________   - LSTM 내부의 게이트 갯수 (4)
dense_1 (Dense)              (None, 10)                210          - output 이 한 번 나갔다가 다시 돌아가서 계산
_________________________________________________________________   - 4*(1+1+10)*10 = 480
dense_2 (Dense)              (None, 1)                 11
=================================================================
Total params: 921
Trainable params: 921
Non-trainable params: 0
_________________________________________________________________
'''

'''
# 3. compile, fitting
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1)

# 4. evaluate, predict
loss=model.evaluate(x,y)
print(loss)

x_pred=np.array([5,6,7]) # (3, )
x_pred=x_pred.reshape(1, 3, 1) # (1, 3, 1) -> ([[[5],[6],[7]]])

result=model.predict(x_pred)
print(result)

# 0.0005654864362441003
# [[8.001797]]
'''