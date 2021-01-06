import numpy as np
from numpy import array

x1=array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
         [5,6,7], [6,7,8], [7,8,9], [8,9,10],
         [9,10,11], [10,11,12],
         [20,30,40],[30,40,50],[40,50,60]])
x2=array([[10,20,30],[20,30,40], [30,40,50],[40,50,60],
         [50,60,70],[60,70,80],[70,80,90],[80,90,100],
         [90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y=array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict=array([55,65,75])
x2_predict=array([65,75,85])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

x1=x1.reshape(x1.shape[0], x1.shape[1],1)
# x2=x2.reshape(13,3,1)
x1_predict=x1_predict.reshape(1,3,1)
x2_predict=x2_predict.reshape(1,3,1)

input1=Input(shape=(3,1)) # x1 데이터의 형태가 좀 더 시계열에 맞음
lstm1=LSTM(100, activation='relu')(input1)
dense1=Dense(100, activation='relu')(lstm1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)

input2=Input(shape=(3,))
dense2=Dense(150, activation='relu')(input2)
dense2=Dense(150, activation='relu')(dense2)
dense2=Dense(150, activation='relu')(dense2)
dense2=Dense(150, activation='relu')(dense2)
dense2=Dense(150, activation='relu')(dense2)
dense2=Dense(150, activation='relu')(dense2)

merge=concatenate([dense1, dense2])
dense3=Dense(200, activation='relu')(merge)
dense3=Dense(200, activation='relu')(dense3)
dense3=Dense(200, activation='relu')(dense3)
dense3=Dense(200, activation='relu')(dense3)
dense3=Dense(200, activation='relu')(dense3)
dense3=Dense(200, activation='relu')(dense3)

dense4=Dense(250, activation='relu')(dense3)
dense4=Dense(250, activation='relu')(dense4)
dense4=Dense(250, activation='relu')(dense4)
dense4=Dense(250, activation='relu')(dense4)
dense4=Dense(250, activation='relu')(dense4)
dense4=Dense(250, activation='relu')(dense4)
output1=Dense(1)(dense4)
model=Model(inputs=[input1, input2], outputs=output1)

early=EarlyStopping(monitor='loss', patience=50, mode='auto')
model.compile(loss='mse', optimizer='adam')
model.fit([x1, x2], y, epochs=140, batch_size=3, callbacks=early)

loss=model.evaluate([x1, x2], y)
y_pred=model.predict([x1_predict, x2_predict])

print(loss)
print(y_pred)

# results - LSTM
# 7.556332111358643
# [[86.66165]]

# results - Dense
# 1.0370432138442993
# [[84.45753]]
