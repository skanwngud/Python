# 다 : 1 mlp 함수형

import numpy as np

x=np.array([range(100), range(301, 401), range(1, 101)])
y=np.array(range(711, 811))

print(x.shape) # (3,100)
print(y.shape) # (100,) - scalar 100

x=np.transpose(x)

print(x.shape) # (100,3)

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape) # (80,3)
print(y_train.shape) # (80,)

input1=Input(shape=(3,))
dense1=Dense(10)(input1)
dense2=Dense(10)(dense1)
dense3=Dense(10)(dense2)
output1=Dense(1)(dense3)
model=Model(inputs=input1, outputs=output1)


# model=Sequential()
# model.add(Dense(10, input_dim=3))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train, epochs=100, batch_size=1, validation_split=0.2)

loss, mae=model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('rmse : ', RMSE(y_test, y_predict))

r2=r2_score(y_test, y_predict)
print('r2 : ', r2)