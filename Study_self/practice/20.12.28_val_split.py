import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

x=np.array(range(1,101))
y=np.array(range(101,201))

x_train=x[:80]
x_test=x[80:]

y_train=y[:80]
y_test=y[80:]

model=Sequential()
model.add(Dense(20, input_dim=1, activation='relu'))
model.add(Dense(20))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train,epochs=100, batch_size=1, validation_split=0.2)

results=model.evaluate(x_test, y_test, batch_size=1)
print(results)

predict=model.predict(y_test)
print(predict)

def RMSE(y_test, predict):
    return np.sqrt(mean_squared_error(y_test, predict))

print(RMSE(y_test, predict))

r2=r2_score(y_test, predict)
print(r2)