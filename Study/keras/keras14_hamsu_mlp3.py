# 1 : 다 함수형

import numpy as np

x=np.array([range(100)])
y=np.array([range(711, 811), range(1, 101), range(201, 301)])
# input data = 1, output data = n 가능


print(x.shape)
print(y.shape) # (3, 100)


x=np.transpose(x)
y=np.transpose(y)

print(x.shape)
print(y.shape) # (100, 3)

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape) # (80, 3)
print(y_train.shape) # (80, 3)

input1=Input(shape=(1,))
aaa=Dense(50)(input1)
aaa=Dense(50)(aaa)
aaa=Dense(50)(aaa)
output=Dense(3)(aaa)
model=Model(inputs=input1, outputs=output)

# model=Sequential()
# model.add(Dense(100, input_dim=1))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(3))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train, epochs=20, batch_size=1, validation_split=0.2, verbose=2)

loss, mae=model.evaluate(x_test, y_test)
print('loss : ', loss) 
print('mae : ', mae)

y_pred=model.predict(x_test)
print(y_pred.shape)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print(y_test.shape)
print(y_pred.shape)

print('rmse : ', RMSE(y_test, y_pred))

r2=r2_score(y_test, y_pred)
print('r2 : ', r2)

print(y_pred)