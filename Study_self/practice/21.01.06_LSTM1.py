import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error

x=np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred=np.array([50,60,70])

x_pred=x_pred.reshape(1,3,1)

print(x.shape)
print(y.shape)
print(x_pred.shape)

input1=Input(shape=(3,1))
lstm1=LSTM(150, activation='relu')(input1)
dense1=Dense(150, activation='relu')(lstm1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
output1=Dense(1)(dense1)
model=Model(input1, output1)

early=EarlyStopping(monitor='loss', patience=50, mode='auto')
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, callbacks=early)

loss=model.evaluate(x,y)
y_pred=model.predict(x_pred)

print(loss)
print(y_pred)

# result
# 0.00012060692097293213
# [[80.450066]]