# 1. data
import numpy as np

x=np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y=np.array([4,5,6,7])


x=x.reshape(4,3,1)

# 2. modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN

model=Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3, 1)))
model.add(SimpleRNN(10, activation='relu', input_length=3, input_dim=1))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
simple_rnn (SimpleRNN)       (None, 10)                120              - (10+1+1)*10
_________________________________________________________________       - 파라미터 아웃값 * (파라미터 아웃값 + 디멘션 값 + 1(바이어스)
dense (Dense)                (None, 20)                220              - default activation = tanh
_________________________________________________________________
dense_1 (Dense)              (None, 10)                210
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11
=================================================================
Total params: 561
Trainable params: 561
Non-trainable params: 0
_________________________________________________________________

# SimpleRNN
# 연속 되는 데이터 중 예측에 크게 연관 되지 않는 데이터들은 무시함
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

