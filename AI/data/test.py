import pandas as pd

data = {
    'A' : ['1', '2', '3', '4'],
    'B' : ['a', 'b', 'c', 'd']
}

df = pd.DataFrame(data)

print(df)

import numpy as np

x = np.array([1., 2., 3.])
y = np.array([1., 2., 3.])

print(x, y)

from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = boston_housing.load_data()
model = Sequential()

model.add(Dense(5, input_shape=(1,), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

# compile, fit
model.compile(loss='mse',metrics=['mse'])
model.fit(x, y, epochs=1000, batch_size=1)

# eval, pred
print(model.evaluate(np.array([5]), np.array([5])))
print(model.predict(np.array([10])))

from tensorflow.keras.utils import to_categorical
from xgboost import XGBRegressor