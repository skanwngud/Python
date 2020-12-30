import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x=np.array(range(1, 101))
y=np.array(range(101, 201))

x_train=np.array(x[:60])
x_val=np.array(x[61:80])
x_test=np.array(x[81:])

y_train=np.array(y[:60])
y_val=np.array(y[61:80])
y_test=np.array(y[81:])

model=Sequential()
model.add(Dense(20, input_dim=1, activation='relu'))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

model.compile(loss='mse', metrics=['mae'], optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

results=model.evaluate(x_test, y_test, batch_size=1)
print('mse, mae : ', results)

y_predict=model.predict(x_test)


from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('RMSE : ', RMSE(y_test, y_predict))
print('mse : ', mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score

r2=r2_score(y_test, y_predict)
print('r2 : ', r2)