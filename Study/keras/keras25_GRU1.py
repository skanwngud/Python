# 1. data
import numpy as np

x=np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y=np.array([4,5,6,7])


x=x.reshape(4,3,1)

# 2. modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU

model=Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3, 1)))
model.add(GRU(10, activation='relu', input_length=3, input_dim=1))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
gru (GRU)                    (None, 10)                390
_________________________________________________________________
dense (Dense)                (None, 20)                220
_________________________________________________________________
dense_1 (Dense)              (None, 10)                210
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11
=================================================================
Total params: 831
Trainable params: 831
Non-trainable params: 0
_________________________________________________________________

- 3 * (노드^2 + 노드 * input_dim + 노드 + 바이어스 ) // 게이트가 3개
- 3 * (1 + 1 + 10) * 10
- default activation = tanh
- 3 * (out^2 * (out+input)+out)

LSTM 에서 cell state 를 뺀 것
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

# LSTM
# 0.0005654864362441003
# [[8.001797]]

# SimpleRNN
# 7.701774302404374e-05
# [[7.9826655]]

# GRU
# 0.007292479742318392
# [[7.9731736]]