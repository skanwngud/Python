import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


x1=np.array([range(100), range(301, 401), range(1, 101)])
y1=np.array([range(711, 811), range(1, 101), range(201, 301)])

x2=np.array([range(101, 201), range(411, 511), range(100,200)])
y2=np.array([range(501,601), range(711, 811), range(100)])

x1=np.transpose(x1)
x2=np.transpose(x2)
y1=np.transpose(y1)
y2=np.transpose(y2)

x1_train, x1_test, y1_train, y1_test=train_test_split(x1, y1, train_size=0.8, random_state=777)
x2_train, x2_test, y2_train, y2_test=train_test_split(x2, y2, train_size=0.8, random_state=777)

input1=Input(shape=3,)
dense1=Dense(10, activation='relu')(input1)
dense1=Dense(9)(dense1)
dense1=Dense(8)(dense1)
dense1=Dense(7)(dense1)

input2=Input(shape=3,)
dense2=Dense(20, activation='relu')(input2)
dense2=Dense(19)(dense2)
dense2=Dense(18)(dense2)
dense2=Dense(17)(dense2)

merge1=concatenate([input1, input2])
hidden1=Dense(30)(merge1)
hidden1=Dense(29)(hidden1)
hidden1=Dense(28)(hidden1)
hidden1=Dense(27)(hidden1)
hidden1=Dense(26)(hidden1)
output1=Dense(3)(hidden1)
model=Model(inputs=[input1, input2], outputs=output1)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=30, batch_size=1, validation_split=0.2)

loss=model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
y_pred=model.predict([x1_test, x2_test])

def rmse(y1_test, y_pred):
    return np.sqrt(mean_squared_error(y1_test, y_pred))

RMSE=rmse(y1_test, y_pred)
r2=r2_score(y1_test, y_pred)

print(loss)
print(RMSE)
print(y_pred)

model.summary()