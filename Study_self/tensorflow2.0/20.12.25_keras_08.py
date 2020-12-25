import numpy as np
from tensorflow.keras.models import Dense
from tensorflow.keras.layers import Sequential

x=np.array(range(0, 101))
y=np.arrat(range(101, 201))

x_train=np.array(x[:60])
x_val=np.array(x[61:80])
x_test=np.array(x[81:])

y_train=np.array(y[:60])
y_val=np.array(y[61:80])
y_test=np.array(y[81:])

model=Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', metrics='r2', optimizer='adam')
model.fit(x_train, y_train, epcohs=100, batch_size=1, validation_data=(x_val, y_val))

model.evaluate(x_test, y_test, batch_size=1)