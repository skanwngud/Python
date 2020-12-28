import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

x=np.array(range(1,101))
y=np.array(range(1,101))

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.8)

model=Sequential()
model.add(Dense(15, input_dim=1, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=120, batch_size=1, validation_split=0.2)

loss=model.evaluate(x_test, y_test, batch_size=1)
pred=model.predict(x_test)

def RMSE(y_test, pred):
    return np.sqrt(mean_squared_error(y_test, pred))

r2=r2_score(y_test, pred)

print(loss)
print(RMSE(y_test, pred))
print(r2)