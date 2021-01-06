import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

a=np.array(range(1, 201))
b=np.array(range(196, 205))
size=6

def split_x(seq, size):
    n=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size)]
        n.append(subset)
    return np.array(n)

dataset=split_x(a, size)


# def split_x(seq, size):
#     n=[]
#     for i in range(len(seq)-size+1):
#         subset=seq[i:(i+size)]
#         n.append(subset)
#     return np.array(n)

# dataset=split_x(a, size)


# print(dataset)

x=dataset[:, :5]
y=dataset[:, -1:]
x1_pred=b[:5]
x2_pred=b[1:6]
x3_pred=b[2:7]
x4_pred=b[3:8]
x5_pred=b[4:9]

# print(x5_pred)
# print(x)
# print(y)
# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val=x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

x1_pred=x1_pred.reshape(1, x1_pred.shape[0], 1)
x2_pred=x2_pred.reshape(1, x2_pred.shape[0], 1)
x3_pred=x3_pred.reshape(1, x3_pred.shape[0], 1)
x4_pred=x4_pred.reshape(1, x4_pred.shape[0], 1)
x5_pred=x5_pred.reshape(1, x5_pred.shape[0], 1)

input1=Input(shape=(5,1))
lstm1=LSTM(500, activation='relu')(input1)
dense1=Dense(500, activation='relu')(lstm1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(500, activation='relu')(dense1)
output1=Dense(1)(dense1)
model=Model(input1, output1)

call=EarlyStopping(monitor='loss', patience=30, mode='auto')
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), callbacks=call)

loss=model.evaluate(x_test, y_test)
y1_pred=model.predict(x1_pred)
y2_pred=model.predict(x2_pred)
y3_pred=model.predict(x3_pred)
y4_pred=model.predict(x4_pred)
y5_pred=model.predict(x5_pred)

print(loss)
print(y1_pred)
print(y2_pred)
print(y3_pred)
print(y4_pred)
print(y5_pred)
