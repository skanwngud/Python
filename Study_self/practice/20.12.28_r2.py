import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x=np.array(range(1,101))
y=np.array(range(101,201))

x_train=x[:60]
x_val=x[60:80]
x_test=x[80:100]

y_train=y[:60]
y_val=y[60:80]
y_test=y[80:100]

model=Sequential()
model.add(Dense(50, input_dim=1, activation='relu'))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

results=model.evaluate(x_test, y_test, batch_size=1)
print('mse, mae : ', results)

y_predict=model.predict(y_test)

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('rmse :', RMSE(y_test, y_predict))
print('mse :', mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)

print('r2 :', r2)