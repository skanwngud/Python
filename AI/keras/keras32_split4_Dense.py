# 데이터 1~100 / size=6 / Dense
# 1,2,3,4,5(x)  6(y)
# prepocessing, callbacks
# predict(5,5) 96, 97, 98, 99, 100 -> 101
#         100, 101, 102, 103, 104 -> 105
# actual predict = 101, 102, 103, 104, 105
# 32_3과 비교

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


# x=x.reshape(x.shape[0], x.shape[1], 1)
x_pred1=x_pred1.reshape(1, x_pred1.shape[0])
x_pred2=x_pred2.reshape(1, x_pred2.shape[0])
x_pred3=x_pred3.reshape(1, x_pred3.shape[0])
x_pred4=x_pred4.reshape(1, x_pred4.shape[0])
x_pred5=x_pred5.reshape(1, x_pred5.shape[0])


early=EarlyStopping(monitor='loss', patience=50, mode='auto')

model=Sequential()
model.add(Dense(150, activation='relu', input_shape=(5,)))
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
model.add(Dense(150, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x, y, epochs=1000, batch_size=5, validation_data=(x, y), callbacks=early)

results=model.evaluate(x, y)
y_pred1=model.predict(x_pred1)
y_pred2=model.predict(x_pred2)
y_pred3=model.predict(x_pred3)
y_pred4=model.predict(x_pred4)
y_pred5=model.predict(x_pred5)

print(results)
print(y_pred1)
print(y_pred2)
print(y_pred3)
print(y_pred4)
print(y_pred5)

# results
# [0.00015044832252897322, 0.009932522661983967]
# [[100.97184]]
# [[101.97089]]
# [[102.97002]]
# [[103.96925]]
# [[104.96847]]