# 데이터 1~100 / size=6 / LSTM
# 1,2,3,4,5(x)  6(y)
# prepocessing, callbacks
# predict(5,5) 96, 97, 98, 99, 100 -> 101
#         100, 101, 102, 103, 104 -> 105
# actual predict = 101, 102, 103, 104, 105

import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

a=np.array(range(1,101))
b=np.array(range(96,105))
size=6

def split_x(seq, size):
    n=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size)]
        n.append(subset)
    return np.array(n)

dataset=split_x(a, size)

x=dataset[:,:-1]
y=dataset[:,-1]
x_pred1=b[:5]
x_pred2=b[1:6]
x_pred3=b[2:7]
x_pred4=b[3:8]
x_pred5=b[4:9]

print(b)
print(b.shape)
print(x.shape)
print(y.shape)
print(x_pred5)
print(x_pred1.shape)

x=x.reshape(x.shape[0], x.shape[1], 1)
# x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)
# x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

# scaler=MinMaxScaler()
# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)
# x_val=scaler.transform(x_val)

# x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
# x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
# x_val=x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

x_pred1=x_pred1.reshape(1, x_pred1.shape[0],1)
x_pred2=x_pred2.reshape(1, x_pred2.shape[0],1)
x_pred3=x_pred3.reshape(1, x_pred3.shape[0],1)
x_pred4=x_pred4.reshape(1, x_pred4.shape[0],1)
x_pred5=x_pred5.reshape(1, x_pred5.shape[0],1)

early=EarlyStopping(monitor='loss', patience=50, mode='auto')

model=Sequential()
model.add(LSTM(350, activation='relu', input_shape=(5,1)))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics='mae')
# model.fit(x_train, y_train, epochs=1500, batch_size=5, validation_data=(x_val, y_val), callbacks=early)

# results=model.evaluate(x_test, y_test)
model.fit(x,y, epochs=100, batch_size=5)
results=model.evaluate(x,y)
y_pred1=model.predict(x_pred1)
y_pred2=model.predict(x_pred2)
y_pred3=model.predict(x_pred3)
y_pred4=model.predict(x_pred4)
y_pred5=model.predict(x_pred5)

print(results)

y_pred=np.array([y_pred1, y_pred2, y_pred3, y_pred4, y_pred5])
print(y_pred)
print(y_pred.shape)
# results
# [0.002582934685051441, 0.045692361891269684]
# [[101.00517]]
# [[101.99776]]
# [[102.98933]]
# [[103.9803]]
# [[104.97096]]