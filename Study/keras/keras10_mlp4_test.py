# 다 : 다

import numpy as np

x=np.array([range(100), range(301, 401), range(1, 101), range(501, 601), range(651, 751)])
y=np.array([range(711, 811), range(101, 201)])

x_pred2=np.array([100, 401, 101, 601, 751]) # (5,)

print(x.shape) # (5, 100)
print(y.shape) # (2, 100)
print(x_pred2.shape)

x=np.transpose(x)
y=np.transpose(y)
# x_pred2=np.transpose(x_pred2) # (5, ) - 1dim data 이기 때문에 transpose 변환식을 쓸 수 없음
x_pred2=x_pred2.reshape(1, 5)
# reshape - 해당 모양으로 배열을 바꿔줌

print(x.shape) # (100, 5)
print(y.shape) # (100, 2)
print(x_pred2.shape) 

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.2, shuffle=True, random_state=66)

print(x_train.shape) # (80, 5)
print(y_train.shape) # (80, 2)

model=Sequential()
model.add(Dense(20, input_dim=5, activation='relu'))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(2))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train, epochs=130, batch_size=1, validation_split=0.2)

loss, mae=model.evaluate(x_test, y_test, batch_size=1)

y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


r2=r2_score(y_test, y_predict)

print(y_predict[:5])
print('rmse : ', RMSE(y_test, y_predict))
print('loss : ', loss)
print('mae : ', mae)
print('r2 : ', r2)

y_pred2=model.predict(x_pred2)
print(y_pred2) # [[811.0607  200.92252]]