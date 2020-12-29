import numpy as np

x=np.array([range(100), range(301, 401), range(1, 101), range(501, 601), range(651, 751)])
y=np.array([range(711, 811), range(101, 201)])

print(x.shape) # (5, 100)
print(y.shape) # (2, 100)

x=np.transpose(x)
y=np.transpose(y)

print(x.shape) # (100, 5)
print(y.shape) # (100, 2)

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.2, shuffle=True, random_state=66)

print(x_train.shape) # (80, 3)
print(y_train.shape) # (80, 3)

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