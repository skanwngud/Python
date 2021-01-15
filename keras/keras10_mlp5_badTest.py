# 다 : 다

import numpy as np

x=np.array([range(100), range(301, 401), range(1, 101), range(100), range(301, 401)])
y=np.array([range(711, 811), range(1, 101)])

x_pred2=np.array([100, 401, 101, 100, 401]) # (5,)

print(x.shape) # (5, 100)
print(y.shape) # (2, 100)
print(x_pred2.shape)

x=np.transpose(x)
y=np.transpose(y)
x_pred2=x_pred2.reshape(1, 5)

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
model.add(Dense(14, input_dim=5))
model.add(Dense(14))
model.add(Dense(14))
model.add(Dense(14))
model.add(Dense(14))
model.add(Dense(14))
model.add(Dense(14))
model.add(Dense(14))
model.add(Dense(14))
model.add(Dense(14))
model.add(Dense(14))
model.add(Dense(14))
model.add(Dense(14))
model.add(Dense(14))
model.add(Dense(2))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train, epochs=36, batch_size=6, validation_split=0.2)

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
print(y_pred2) 

# r2 :  0.3726407271921447 / relu (natch_size=45, layer = 5)
# r2 :  0.21675415358324146 / linear (batch_size=50, layer = 5)
# r2 :  0.4548206209082694 / linear (batch_size=36, layer > 10)
