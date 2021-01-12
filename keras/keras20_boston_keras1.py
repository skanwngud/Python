# EarlyStopping 을 적용하지 않은 모델

from tensorflow.keras.datasets import boston_housing # using this file

import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

(train_data, train_target), (test_data, test_target)=boston_housing.load_data()

x_train=train_data
y_train=train_target

x_test=test_data
y_test=test_target

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_val=scaler.transform(x_val)
x_test=scaler.transform(x_test)

input1=Input(shape=13)
dense1=Dense(150, activation='relu')(input1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
output1=Dense(1, activation='relu')(dense1)
model=Model(inputs=input1, outputs=output1)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=200, batch_size=16, validation_data=(x_val, y_val))

loss, mae=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

rmse=RMSE(y_test, y_pred)
r2=r2_score(y_test, y_pred)

print('mse : ', loss)
print('mae : ', mae)
print('rmse : ', rmse)
print('r2 : ', r2)

# results
# mse :  11.866742134094238
# mae :  2.380223035812378
# rmse :  3.444813860678402
# r2 :  0.8574460125795427