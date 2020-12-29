import numpy as np

x=np.array([[1,2,3,4,5,6,7,8,9,10],
            [11,12,13,14,15,16,17,18,19,20]])
# x=np.array([[1,2], [3,4], [5,6], [7,8], [9,10],
#            [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]])
y=np.array([1,2,3,4,5,6,7,8,9,10])

x=np.transpose(x)
# x=x.reshape(10,2)

print(x)
print(x.shape)
# (10,) - scarlar 10
# (2, 10) - vector 2, scarlar 10

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense - 위의 방식에 비해 tensorflow 를 한 번 구동시키고 불러오기 때문에 속도가 느림

model=Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x,y, epochs=100, batch_size=1, validation_split=0.2)

loss, mae=model.evaluate(x,y)
print('loss : ', loss)
print('mae : ', mae)

y_predict=model.predict(x)
# print(y_predict)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(y, y_predict):
    return np.sqrt(mean_squared_error(y, y_predict))

print('rmse : ', RMSE(y, y_predict))

r2=r2_score(y, y_predict)
print('r2 : ', r2)