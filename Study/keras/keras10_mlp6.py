# 1 : 다

import numpy as np

x=np.array([range(100)])
y=np.array([range(711, 811), range(1, 101), range(201, 301)])
# input data = 1, output data = n 가능
x_pred=np.array([101, 811, 301])

print(x.shape)
print(y.shape) # (3, 100)

x=np.transpose(x)
y=np.transpose(y)
x_pred=x_pred.reshape(3,1)

print(x.shape)
print(y.shape) # (100, 3)

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape) # (80, 3)
print(y_train.shape) # (80, 3)

model=Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(3))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train, epochs=20, batch_size=1, validation_split=0.2, verbose=2)

loss, mae=model.evaluate(x_test, y_test)
print('loss : ', loss) 
print('mae : ', mae)

y_pred=model.predict(x_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print('rmse : ', RMSE(y_test, y_pred))

r2=r2_score(y_test, y_pred)
print('r2 : ', r2)

y_pred2=model.predict(x_pred)
print(y_pred2)