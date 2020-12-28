import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_train=([1,2,3,4,5])
y_train=([1,2,3,4,5])

x_test=([10,11,12])
y_test=([10,11,12])

model=Sequential()

a=x_train
b=y_train
c=x_test
d=y_test

model.add(Dense(50, input_dim=1, activation='relu'))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

def skanwngud(a,b,c,d):
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.fit(a,b, epochs=100, batch_size=1)

    result=model.evaluate(c,d, batch_size=1)

    predict=model.predict(c)

    return print(result, '\n', predict)

skanwngud(a,b,c,d)
