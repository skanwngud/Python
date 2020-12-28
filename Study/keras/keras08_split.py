from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
from numpy import array

#1. data
x=np.array(range(1,101))
# x=np.array(range(100))
y=np.array(range(101,201))

x_train=x[:60] # 0번째부터 59번째 수까지 = 1~60
x_val=x[60:80] # 61번째부터 79번째 수까지 = 61~80
x_test=x[80:] # 81번째부터 끝까지 = 81~100
# 슬라이싱

y_train=y[:60]
y_val=y[60:80]
y_test=y[80:]

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
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))
# validation 데이터를 위에서 줬으므로 validation_split 이 아닌 validation_data 로 씀

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
