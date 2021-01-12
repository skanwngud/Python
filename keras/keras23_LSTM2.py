# input_shape / input_length / input_dim

# 1. data
import numpy as np

x=np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y=np.array([4,5,6,7])


x=x.reshape(4,3,1)

# 2. modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model=Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3, 1)))
model.add(LSTM(10, activation='relu', input_length=3, input_dim=1))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

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
