from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
from numpy import array

#1. data
x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([1,2,3,4,5,6,7,8,9,10])

x_test=array([11,12,13,14,15])
y_test=array([11,12,13,14,15])

x_pred=array([16,17,18])

#2. modeling
model=Sequential()
model.add(Dense(50, input_dim=1, activation='relu'))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

#3. compile, training
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#4. evaluate, predict
results=model.evaluate(x_test, y_test, batch_size=1)
print("mse, mae : ", results)

y_predict=model.predict(x_test)
# print('y_predict : ', y_predict)

# np.sqrt(results[0]) : same as RMSE

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))
# print('mse : ', mean_squared_error(y_test, y_predict))
print('mse : ', mean_squared_error(y_predict, y_test))

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print("R2 : ", r2)