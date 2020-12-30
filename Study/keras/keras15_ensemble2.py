import numpy as np

x1=np.array([range(100), range(301, 401), range(1, 101)])
x2=np.array([range(101, 201), range(411, 511), range(100,200)])
y1=np.array([range(711, 811), range(1, 101), range(201, 301)])

x1=np.transpose(x1)
x2=np.transpose(x2)
y1=np.transpose(y1)

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test=train_test_split(x1, y1, shuffle=False, train_size=0.8)
x2_train, x2_test=train_test_split(x2, shuffle=False, train_size=0.8)
# x2_train, x2_test, y1_train, y1_test=train_test_split(x2, y1, shuffle=False, train_size=0.8)
# x1_train, x1_test, x2_train, x2_test, y1_trian, y1_test=train_test_split(x1, x2, y1, shuffle=False, train_size=0.8)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# model 1
input1=Input(shape=(3,))
dense1=Dense(10, activation='relu')(input1)
dense1=Dense(10, activation='relu')(input1)
dense1=Dense(10, activation='relu')(input1)
dense1=Dense(10, activation='relu')(input1)
dense1=Dense(5, activation='relu')(dense1)
# output1=Dense(3)(dense1)

# model 2
input2=Input(shape=(3,))
dense2=Dense(10, activation='relu')(input2)
dense2=Dense(15, activation='relu')(dense2)
dense2=Dense(15, activation='relu')(dense2)
dense2=Dense(15, activation='relu')(dense2)
dense2=Dense(5, activation='relu')(dense2)
# output2=Dense(3)(dense2)

## 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate
# from keras.layers import concatenate

merge1=concatenate([dense1, dense2])
middle1=Dense(30)(merge1)
middle1=Dense(10)(middle1)
middle1=Dense(10)(middle1)

output1=Dense(30)(middle1)
output1=Dense(7)(output1)
output1=Dense(3)(output1)

model=Model(inputs=[input1, input2], outputs=output1)
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y1_train, epochs=130, batch_size=1, validation_split=0.2, verbose=2)

loss=model.evaluate([x1_test, x2_test], y1_test, batch_size=1)

# print('model.metrics_name : ', model.metrics_names)

y1_predict=model.predict([x1_test, x2_test])

# print('y1_pred : \n', y1_predict)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE_1(y1_test, y1_predict):
    return np.sqrt(mean_squared_error(y1_test, y1_predict))

RMSE1=RMSE_1(y1_test, y1_predict)
r2_1=r2_score(y1_test, y1_predict)

print('loss : ', loss)
print('rmse1 : ', RMSE1)
print('r2_1 : ', r2_1)