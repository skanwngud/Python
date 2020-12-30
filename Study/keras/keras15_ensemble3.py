import numpy as np

x1=np.array([range(100), range(301, 401), range(1, 101)])
y1=np.array([range(711, 811), range(1, 101), range(201, 301)])

x2=np.array([range(101, 201), range(411, 511), range(100,200)])
y2=np.array([range(501,601), range(711, 811), range(100)])

y3=np.array([range(601,701), range(811, 911), range(1100, 1200)])


x1=np.transpose(x1)
x2=np.transpose(x2)
y1=np.transpose(y1)
y2=np.transpose(y2)
y3=np.transpose(y3)


from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test=train_test_split(x1, y1, shuffle=False, train_size=0.8)
x2_train, x2_test, y2_train, y2_test=train_test_split(x2, y2, shuffle=False, train_size=0.8)
y3_train, y3_test=train_test_split(y3, shuffle=False, train_size=0.8)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# model 1
input1=Input(shape=(3,))
dense1=Dense(10, activation='relu')(input1)
dense1=Dense(15)(dense1)
dense1=Dense(15)(dense1)
dense1=Dense(15)(dense1)
dense1=Dense(15)(dense1)
dense1=Dense(15)(dense1)
dense1=Dense(15)(dense1)

# model 2
input2=Input(shape=(3,))
dense2=Dense(10, activation='relu')(input2)
dense2=Dense(25)(dense2)
dense2=Dense(25)(dense2)
dense2=Dense(25)(dense2)
dense2=Dense(25)(dense2)
dense2=Dense(25)(dense2)
dense2=Dense(25)(dense2)
dense2=Dense(25)(dense2)

## 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate
# from keras.layers import concatenate

merge1=concatenate([dense1, dense2])
middle1=Dense(30)(merge1)
middle1=Dense(10)(middle1)
middle1=Dense(10)(middle1)
middle1=Dense(10)(middle1)
middle1=Dense(10)(middle1)
middle1=Dense(10)(middle1)

## 모델 분기 1
output1=Dense(30)(middle1)
output1=Dense(7)(output1)
output1=Dense(7)(output1)
output1=Dense(7)(output1)
output1=Dense(3)(output1)

## 모델 분기 2
output2=Dense(15)(middle1)
output2=Dense(7)(output2)
output2=Dense(7)(output2)
output2=Dense(7)(output2)
output2=Dense(3)(output2)

## 모델 분기 3
output3=Dense(20)(middle1)
output3=Dense(20)(output3)
output3=Dense(15)(output3)
output3=Dense(15)(output3)
output3=Dense(3)(output3)

# 모델 선언
model=Model(inputs=[input1, input2], outputs=[output1, output2, output3])

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train], epochs=80, batch_size=1, validation_split=0.2, verbose=3)

loss=model.evaluate([x1_test, x2_test], [y1_test, y2_test, y3_test], batch_size=1)

print('model.metrics_name : ', model.metrics_names)
# model.metrics_name :  ['loss', 'dense_11_loss', 'dense_15_loss', 'dense_11_mae', 'dense_15_mae']

y1_predict, y2_predict, y3_predict=model.predict([x1_test, x2_test])

print('y1_pred : \n', y1_predict)
print('y2_pred : \n', y2_predict)
print('y3_pred : \n', y3_predict)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE_1(y1_test, y1_predict):
    return np.sqrt(mean_squared_error(y1_test, y1_predict))

def RMSE_2(y2_test, y2_predict):
    return np.sqrt(mean_squared_error(y2_test, y2_predict))

def RMSE_3(y3_test, y3_predict):
    return np.sqrt(mean_squared_error(y3_test, y3_predict))

RMSE1=RMSE_1(y1_test, y1_predict)
RMSE2=RMSE_2(y2_test, y2_predict)
RMSE3=RMSE_3(y3_test, y3_predict)
RMSE=(RMSE1+RMSE2+RMSE3)/3

r2_1=r2_score(y1_test, y1_predict)
r2_2=r2_score(y2_test, y2_predict)
r2_3=r2_score(y3_test, y3_predict)
r2=(r2_1+r2_2+r2_3)/3

print('rmse1 : ', RMSE_1(y1_test, y1_predict))
print('rmse2 : ', RMSE_2(y2_test, y2_predict))
print('rmse3 : ', RMSE_3(y3_test, y3_predict))
print('rmse_sum : ', RMSE)

print('r2_1 : ', r2_1)
print('r2_2 : ', r2_2)
print('r2_3 : ', r2_3)
print('r2_sum : ', r2)