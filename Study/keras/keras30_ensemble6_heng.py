# 데이터의 크기(데이터의 행)가 서로 다른 경우 - 돌아가지 않으니 데이터의 행을 맞춰야한다

import numpy as np
from numpy import array

x1=array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
         [5,6,7], [6,7,8], [7,8,9], [8,9,10],
         [9,10,11], [10,11,12]])
x2=array([[10,20,30],[20,30,40], [30,40,50],[40,50,60],
         [50,60,70],[60,70,80],[70,80,90],[80,90,100],
         [90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y1=array([4,5,6,7,8,9,10,11,12,13])
y2=array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict=array([55,65,75])
x2_predict=array([65,75,85]) # (3, ) - > (1, 3) - > (1, 3, 1) / 기본, Dense, LSTM

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

x1=x1.reshape(x1.shape[0],x1.shape[1],1)
x2=x2.reshape(x2.shape[0],x2.shape[1],1)
x1_predict=x1_predict.reshape(1,3,1)
x2_predict=x2_predict.reshape(1,3,1)

# print(x1.shape) # 10, 3
# print(x2.shape) # 13, 3
# print(y1.shape) # 10,
# print(y2.shape) # 13,
# print(x1_predict.shape)
# print(x2_predict.shape)

input1=Input(shape=(3,1))
lstm1=LSTM(100, activation='relu')(input1)
dense1=Dense(100, activation='relu')(lstm1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)

input2=Input(shape=(3,1))
lstm2=LSTM(150, activation='relu')(input2)
dense2=Dense(150, activation='relu')(lstm2)
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

dense5=Dense(300, activation='relu')(dense3)
dense5=Dense(300, activation='relu')(dense5)
dense5=Dense(300, activation='relu')(dense5)
dense5=Dense(300, activation='relu')(dense5)
dense5=Dense(300, activation='relu')(dense5)
output2=Dense(1)(dense5)

model=Model(inputs=[input1, input2], outputs=[output1, output2])

early=EarlyStopping(monitor='loss', patience=50, mode='auto')
model.compile(loss='mse', optimizer='adam')
model.fit([x1, x2], [y1, y2], epochs=140, batch_size=3, callbacks=early)

loss=model.evaluate([x1, x2], [y1, y2])
y_pred=model.predict([x1_predict, x2_predict])

print(loss)
print(y_pred)

# results
# ValueError: Data cardinality is ambiguous:
#   x sizes: 10, 13
#   y sizes: 10, 13
# Please provide data which shares the same first dimension.
