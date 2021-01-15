# sklearn / LSTM - regression

import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping


dataset=load_boston()

x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1],1 )
x_val=x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

# y_test=y_test.reshape(1, y_test.shape[0])
print(x_train.shape) # 323, 13, 1
print(x_test.shape)
print(x_val.shape)

model=Sequential()
model.add(LSTM(256, activation='relu', input_shape=(13, 1)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

early=EarlyStopping(monitor='loss', patience=30, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), callbacks=early)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

rmse=RMSE(y_test, y_pred)
r2=r2_score(y_test, y_pred)

print('mse , mae : ', loss)
print('rmse : ', rmse)
print('r2 : ', r2)

# results - 튜닝 전
# mse , mae :  [18.189115524291992, 3.100198745727539]
# rmse :  4.264869931709073
# r2 :  0.7823824321680437

# results - after tune
# mse , mae :  [6.426943778991699, 1.8864681720733643]
# rmse :  2.5351418882196364
# r2 :  0.9231069809392667