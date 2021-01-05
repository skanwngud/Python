import numpy as np

# 1. data
x=np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred=np.array([50,60,70])
# y_pred = 80

# 2. modeling
x=x.reshape(13,3,1)
x_pred=x_pred.reshape(1,3, 1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, SimpleRNN, GRU
from tensorflow.keras.callbacks import EarlyStopping

# x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)
# x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

# model=Sequential()
# model.add(LSTM(256, activation='relu', input_shape=(3,1)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))

input1=Input(shape=(3, 1))
dense1=GRU(220, activation='relu')(input1)
dense1=Dense(220, activation='relu')(dense1)
dense1=Dense(220, activation='relu')(dense1)
output1=Dense(1)(dense1)
model=Model(inputs=input1, outputs=output1)

early=EarlyStopping(monitor='loss', patience=50, mode='auto')
model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_val, y_val), callbacks=early)

# loss=model.evaluate(x_test, y_test)
model.fit(x,y, epochs=1000, batch_size=1, callbacks=early)
loss=model.evaluate(x, y)
y_pred=model.predict(x_pred)

print(loss)
print(y_pred)

# result (LSTM)
# 0.07782885432243347
# [[80.32742]]

# result (Simple RNN)
# 0.3722700774669647
# [[79.76819]]

# result (GRU)
# 2.439350128173828
# [[82.85989]]