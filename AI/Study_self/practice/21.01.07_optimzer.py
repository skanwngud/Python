import numpy as np

from tensorflow.keras.optimizers import SGD, Adam

x=np.array([1,2,3])
y=np.array([1,2,3])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(1))

model.compile(loss='mse', optimizer='sgd')
model.fit(x,y, epochs=100)

loss=model.evaluate(x,y)
pred=model.predict(x)

print(loss)
print(pred)